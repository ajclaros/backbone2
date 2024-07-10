import os
import orjson
from tqdm import tqdm
import argparse
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Generate embeddings for scientific papers")
parser.add_argument("--data_path", type=str, required=False, help="Path to the cleaned data directory", default="../../cleaned")
# parser.add_argument("--output_path", type=str, required=True, help="Path to save Parquet files")
parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased", help="Name of the HuggingFace model to use")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for embedding generation")
parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(model_name=args.model_name, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': False})

def process_document(doc):
    paper_id = doc['paper_id']
    abstract_embed = embeddings.embed_query(doc.get('cleaned_abstract', '').lower())

    body_embeds = []
    for section in doc.get('cleaned_body', []):
        body_embeds.append(embeddings.embed_query(section['text'].lower()))

    return {
        'paper_id': paper_id,
        'abstract_embedding': abstract_embed,
        'body_embeddings': body_embeds,
        # 'original_json': orjson.dumps(doc).decode()  # Store original JSON as string
    }

def process_file(file_path):
    with open(file_path, 'r') as f:
        documents = [orjson.loads(line) for line in f]

    processed_docs = []
    for doc in tqdm(documents, desc=f"Processing {os.path.basename(file_path)}"):
        processed_docs.append(process_document(doc))

    return processed_docs

def save_to_parquet(data, output_file):
    table = pa.Table.from_pylist(data)
    pq.write_table(table, output_file)

def main():
    for year in os.listdir(args.data_path):
        year_path = os.path.join(args.data_path, year)
        if not os.path.isdir(year_path):
            continue

        for field in os.listdir(year_path):
            field_path = os.path.join(year_path, field)
            if not os.path.isdir(field_path):
                continue
            print(f"Processing year {year}, field {field}")
            # save in the same directory as the cleaned data
            output_file = os.path.join(field_path, f"{year}_{field}.parquet")

            if os.path.exists(output_file):
                print(f"Skipping {year}_{field}, Parquet file already exists.")
                continue

            files = [os.path.join(field_path, f) for f in os.listdir(field_path) if f.endswith('.jsonl')]

            all_processed_docs = []
            with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
                for processed_docs in executor.map(process_file, files):
                    all_processed_docs.extend(processed_docs)

            save_to_parquet(all_processed_docs, output_file)
            print(f"Saved {output_file}")

if __name__ == "__main__":
    main()
