import re

def extract_page_number(filename: str):
    match = re.search(r'page[_\-]?(\d+)', filename.lower())
    return int(match.group(1)) if match else None

import re

def normalize_table_rows(text: str) -> str:
    """
    Converts table summary rows like:
    'Active clients: Dec 31 2025 = 1949, Sep 30 2025 = 1896'

    into:
    'Active clients Dec 31 2025 = 1949
     Active clients Sep 30 2025 = 1896'
    """

    normalized_lines = []

    for line in text.split("\n"):

        if ":" in line and "=" in line:
            label, values = line.split(":", 1)

            matches = re.findall(r"([A-Za-z0-9 ]+?)\s*=\s*([0-9,.%]+)", values)

            for m in matches:
                normalized_lines.append(f"{label.strip()} {m[0].strip()} = {m[1]}")

        else:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)