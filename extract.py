import json
import os
import multiprocessing
from typing import Dict, List, Tuple, Iterator
from functools import partial
from tqdm import tqdm
import psutil

def load_jsonl(file_path: str) -> Iterator[Dict]:
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)

def clean_text(text: str, formula_dict: Dict, citation_dict: Dict) -> Tuple[str, Dict, Dict]:
    formula_counter = 1
    citation_counter = 1
    formula_lookup = {}
    citation_lookup = {}

    for formula_id, formula in formula_dict.items():
        placeholder = f"[FORMULA_{formula_counter}]"
        text = text.replace(f"{{{{formula:{formula_id}}}}}", placeholder)
        formula_lookup[placeholder] = formula
        formula_counter += 1

    for citation_id, citation in citation_dict.items():
        placeholder = f"[CITATION_{citation_counter}]"
        text = text.replace(f"{{{{cite:{citation_id}}}}}", placeholder)
        citation_lookup[placeholder] = citation
        citation_counter += 1

    return text, formula_lookup, citation_lookup

def process_paper(paper: Dict) -> Dict:
    cleaned_abstract, abstract_formula_lookup, abstract_citation_lookup = clean_text(
        paper['abstract']['text'],
        paper['ref_entries'],
        paper['bib_entries']
    )

    cleaned_body = []
    for section in paper['body_text']:
        cleaned_section, section_formula_lookup, section_citation_lookup = clean_text(
            section['text'],
            paper['ref_entries'],
            paper['bib_entries']
        )
        cleaned_body.append({
            'section': section['section'],
            'text': cleaned_section,
            'formula_lookup': section_formula_lookup,
            'citation_lookup': section_citation_lookup
        })

    return {
        'paper_id': paper['paper_id'],
        'cleaned_abstract': cleaned_abstract,
        'abstract_formula_lookup': abstract_formula_lookup,
        'abstract_citation_lookup': abstract_citation_lookup,
        'cleaned_body': cleaned_body
    }

def process_chunk(papers: List[Dict]) -> List[Dict]:
    return [process_paper(paper) for paper in papers]

def chunked_parallel_process(papers_iterator: Iterator[Dict], chunk_size: int, num_processes: int) -> Iterator[Dict]:
    with multiprocessing.Pool(processes=num_processes) as pool:
        while True:
            chunk = list(itertools.islice(papers_iterator, chunk_size))
            if not chunk:
                break
            for result in pool.imap(process_paper, chunk):
                yield result

def process_year_field(year: int, field: str, source_dir: str, output_dir: str, num_processes: int, chunk_size: int, max_memory_percent: float):
    year_dir = os.path.join(source_dir, str(year))
    output_year_dir = os.path.join(output_dir, str(year), field)
    os.makedirs(output_year_dir, exist_ok=True)

    output_file = os.path.join(output_year_dir, f"cleaned_{field}_{year}.jsonl")

    def paper_generator():
        for file_name in os.listdir(year_dir):
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(year_dir, file_name)
                for paper in load_jsonl(file_path):
                    if paper['discipline'] == field:
                        yield paper

    total_papers = sum(1 for _ in paper_generator())

    with open(output_file, 'w') as out_file:
        for processed_paper in tqdm(
            chunked_parallel_process(paper_generator(), chunk_size, num_processes),
            total=total_papers,
            desc=f"Processing {field} papers for {year}"
        ):
            json.dump(processed_paper, out_file)
            out_file.write('\n')

            # Check memory usage and adjust if necessary
            if psutil.virtual_memory().percent > max_memory_percent:
                num_processes = max(1, num_processes - 1)
                chunk_size = max(1, chunk_size // 2)
                print(f"Memory usage high. Adjusting to {num_processes} processes and chunk size {chunk_size}")

def main(source_dir: str, output_dir: str, years: List[int], fields: List[str], num_processes: int, chunk_size: int, max_memory_percent: float):
    for year in years:
        for field in fields:
            process_year_field(year, field, source_dir, output_dir, num_processes, chunk_size, max_memory_percent)

if __name__ == "__main__":
    source_directory = "/path/to/uncompressed/data"
    output_directory = "/path/to/processed/data"
    years_to_process = [2021, 2022, 2023]
    fields_to_process = ["Computer Science", "Physics", "Mathematics"]
    num_processes = multiprocessing.cpu_count() // 2  # Use half of available CPU cores
    chunk_size = 100  # Process 100 papers at a time
    max_memory_percent = 80.0  # Adjust processes if memory usage exceeds 80%

    main(source_directory, output_directory, years_to_process, fields_to_process, num_processes, chunk_size, max_memory_percent)
