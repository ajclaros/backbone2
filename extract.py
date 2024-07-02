import json
import os
from typing import Dict, List, Tuple

def load_jsonl(file_path: str) -> List[Dict]:
    """Load a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def filter_papers_by_field(papers: List[Dict], field: str) -> List[Dict]:
    """Filter papers by the specified field."""
    return [paper for paper in papers if paper['discipline'] == field]

def clean_text(text: str, formula_dict: Dict, citation_dict: Dict) -> Tuple[str, Dict, Dict]:
    """Clean the text by replacing formulas and citations with placeholders."""
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
    """Process a single paper by cleaning its text content."""
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

def process_papers_for_year_and_field(year: int, field: str, source_dir: str, output_dir: str):
    """Process papers for a specific year and field, and save the results."""
    year_dir = os.path.join(source_dir, str(year))
    output_year_dir = os.path.join(output_dir, str(year))
    os.makedirs(output_year_dir, exist_ok=True)

    for file_name in os.listdir(year_dir):
        if file_name.endswith('.jsonl'):
            input_path = os.path.join(year_dir, file_name)
            output_path = os.path.join(output_year_dir, f"cleaned_{field}_{file_name}")

            papers = load_jsonl(input_path)
            filtered_papers = filter_papers_by_field(papers, field)
            processed_papers = [process_paper(paper) for paper in filtered_papers]

            with open(output_path, 'w') as out_file:
                for paper in processed_papers:
                    json.dump(paper, out_file)
                    out_file.write('\n')

# Example usage
field_to_process = "Physics"
source_directory = "../.."
output_directory = f"../../cleaned/{field_to_process.lower()}"
year_to_process = "00"

process_papers_for_year_and_field(year_to_process, field_to_process, source_directory, output_directory)
