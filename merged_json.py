import os
import json
from collections import defaultdict


TEXT_ROOT = "extracted_assets"
IMAGE_SUMMARY_ROOT = "output"
MERGED_ROOT = "merged_output"


def load_text_pages(pdf_name):
    """
    Load clean extracted text pages.
    Returns: dict {page_number: text}
    """
    text_path = os.path.join(TEXT_ROOT, pdf_name, "text")

    text_pages = {}

    if not os.path.exists(text_path):
        return text_pages

    for file in os.listdir(text_path):
        if not file.endswith(".json"):
            continue

        file_path = os.path.join(text_path, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        page_number = data.get("page")
        #text = data.get("text", "")

        text_pages[page_number] = {
            "text": data.get("text", ""),
            "keywords": data.get("keywords", [])
        }

        #text_pages[page_number] = text

    return text_pages


def load_visual_summaries(pdf_name):
    """
    Load image/table/vector summaries.
    Returns: dict {page_number: [visual_entries]}
    """

    visual_root = os.path.join(IMAGE_SUMMARY_ROOT, pdf_name)

    visual_pages = defaultdict(list)

    if not os.path.exists(visual_root):
        return visual_pages

    for asset_type in os.listdir(visual_root):

        asset_folder = os.path.join(visual_root, asset_type)

        if not os.path.isdir(asset_folder):
            continue

        for file in os.listdir(asset_folder):

            if not file.endswith(".json"):
                continue

            file_path = os.path.join(asset_folder, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            page_number = data.get("page_number")

            visual_entry = {
                "type": asset_type,
                "image_title": data.get("image_title"),
                "image_keywords": data.get("image_keywords"),
                "summary": data.get("summary"),  # Used temporarily for text merge
                "table_rows": data.get("table_rows", []),
                "confidence_score": data.get("confidence_score"),
                "image_citation" : data.get("image_citation")
            }

            visual_pages[page_number].append(visual_entry)

    return visual_pages


def merge_pdf_content(pdf_name):

    text_pages = load_text_pages(pdf_name)
    visual_pages = load_visual_summaries(pdf_name)

    all_page_numbers = set(text_pages.keys()) | set(visual_pages.keys())

    merged_pages = []

    for page in sorted(all_page_numbers):

        #base_text = text_pages.get(page, "").strip()
        page_data = text_pages.get(page, {})
        base_text = page_data.get("text", "").strip()
        keywords = page_data.get("keywords", [])
        visuals = visual_pages.get(page, [])


        # 🔥 Append visual summaries into text
        for v in visuals:

            table_rows = v.get("table_rows", [])
            rows_text = "\n".join(table_rows)

            summary_block = (
                f"\n\n[VISUAL CONTENT - {v.get('type', '').upper()}]\n"
                f"Title: {v.get('image_title', '')}\n"
                f"Summary: {v.get('summary', '')}\n"
                f"{rows_text}"
            )
            base_text += summary_block

        # 🔥 Remove summary from retained visual metadata
        cleaned_visuals = []
        for v in visuals:
            cleaned_visuals.append({
                "type": v.get("type"),
                "image_title": v.get("image_title"),
                "image_keywords": v.get("image_keywords"),
                #"confidence_score": v.get("confidence_score"),
                "image_citation": v.get("image_citation")
            })

        merged_pages.append({
            "pdf": pdf_name,
            "page": page,
            "text": base_text,
            "keywords": keywords,
            "visual_summaries": cleaned_visuals
        })

    return merged_pages


def run_merge():

    os.makedirs(MERGED_ROOT, exist_ok=True)

    for pdf_name in os.listdir(TEXT_ROOT):

        pdf_path = os.path.join(TEXT_ROOT, pdf_name)

        if not os.path.isdir(pdf_path):
            continue

        merged_content = merge_pdf_content(pdf_name)

        output_path = os.path.join(MERGED_ROOT, f"{pdf_name}_merged.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_content, f, indent=2, ensure_ascii=False)

        print(f"Merged saved → {output_path}")


if __name__ == "__main__":
    run_merge()