import json
import os
import multiprocessing
from typing import Dict, List, Tuple
from functools import partial
from tqdm import tqdm

def load_jsonl(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

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

def process_papers_parallel(papers: List[Dict], num_processes: int) -> List[Dict]:
    with multiprocessing.Pool(processes=num_processes) as pool:
        processed_papers = list(tqdm(
            pool.imap(process_paper, papers),
            total=len(papers),
            desc="Processing papers"
        ))
    return processed_papers

def process_year_field(year: int, field: str, source_dir: str, output_dir: str, num_processes: int):
    year_dir = os.path.join(source_dir, str(year))
    output_year_dir = os.path.join(output_dir, str(year), field)
    os.makedirs(output_year_dir, exist_ok=True)

    all_papers = []
    for file_name in os.listdir(year_dir):
        if file_name.endswith('.jsonl'):
            file_path = os.path.join(year_dir, file_name)
            papers = load_jsonl(file_path)
            all_papers.extend([p for p in papers if p['discipline'] == field])

    processed_papers = process_papers_parallel(all_papers, num_processes)

    output_file = os.path.join(output_year_dir, f"cleaned_{field}_{year}.jsonl")
    with open(output_file, 'w') as out_file:
        for paper in processed_papers:
            json.dump(paper, out_file)
            out_file.write('\n')

def main(source_dir: str, output_dir: str, years: List[int], fields: List[str], num_processes: int):
    for year in years:
        for field in fields:
            print(f"Processing {field} papers for {year}")
            process_year_field(year, field, source_dir, output_dir, num_processes)

source_directory = "../.."
output_directory = "../../cleaned"
years_to_process = ["00", "01", "02", "03"]
fields_to_process = ["Computer Science", "Physics", "Mathematics"]
num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
main(source_directory, output_directory, years_to_process, fields_to_process, num_processes)
