import os
import json
from config.settings import INPUT_FOLDER
from processors.pdf_content_detector import detect_pdf_content


OUTPUT_FOLDER = "pdf_metadata"


def generate_pdf_metadata():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for file in os.listdir(INPUT_FOLDER):

        if not file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(INPUT_FOLDER, file)

        print(f"Processing PDF: {file}")

        metadata = detect_pdf_content(pdf_path)

        output_path = os.path.join(
            OUTPUT_FOLDER,
            file.replace(".pdf", "_metadata.json")
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Saved: {output_path}")