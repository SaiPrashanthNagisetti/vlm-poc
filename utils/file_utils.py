import re

def extract_page_number(filename: str):
    match = re.search(r'page[_\-]?(\d+)', filename.lower())
    return int(match.group(1)) if match else None