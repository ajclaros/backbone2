import os
import orjson
import numpy as np
from tqdm import tqdm
import argparse
from langchain_community.embeddings import HuggingFaceEmbeddings

parser = argparse.ArgumentParser(description="Generate embeddings for scientific papers")
parser.add_argument("--data_path", type=str, required=False, help="Path to the cleaned data directory", default="../../cleaned")
parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased", help="Name of the HuggingFace model to use")

args = parser.parse_args()

device = "cuda"
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
    }

def process_file(file_path, output_dir):
    with open(file_path, 'r') as f:
        documents = [orjson.loads(line) for line in f]

    for doc in tqdm(documents, desc=f"Processing {os.path.basename(file_path)}"):
        processed_doc = process_document(doc)
        save_as_npy(processed_doc, output_dir)

def save_as_npy(data, output_dir):
    paper_id = data['paper_id']
    output_file = os.path.join(output_dir, f"{paper_id}.npy")
    np.save(output_file, data)

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

            # Create output directory for embeddings
            output_dir = os.path.join(field_path, 'embeddings')
            os.makedirs(output_dir, exist_ok=True)

            files = [os.path.join(field_path, f) for f in os.listdir(field_path) if f.endswith('.jsonl')]

            for file in files:
                process_file(file, output_dir)

            print(f"Finished processing {year}_{field}")

if __name__ == "__main__":
    main()
