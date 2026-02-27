import os
import json
from config.settings import INPUT_FOLDER
from utils.file_utils import extract_page_number
from services.image_analyzer import analyze_image


OUTPUT_ROOT = "output"


def process_images():

    for folder_name in os.listdir(INPUT_FOLDER):

        input_folder_path = os.path.join(INPUT_FOLDER, folder_name)

        if not os.path.isdir(input_folder_path):
            continue

        # Create matching output folder
        output_folder_path = os.path.join(OUTPUT_ROOT, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        for image_file in os.listdir(input_folder_path):

            if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(input_folder_path, image_file)

            print(f"Processing: {image_path}")

            page_number = extract_page_number(image_file)

            model_output, usage = analyze_image(image_path)

            final_metadata = {
                "file_name": folder_name,
                "page_number": page_number,
                "image_title": model_output.get("image_title"),
                "image_keywords": model_output.get("image_keywords"),
                "summary": model_output.get("summary"),
                "confidence_score": model_output.get("confidence_score"),

                "token_usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens")
                }
            }

            # Create JSON file name same as image
            json_filename = os.path.splitext(image_file)[0] + ".json"
            json_output_path = os.path.join(output_folder_path, json_filename)

            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(final_metadata, f, indent=4)

            print(f"Saved: {json_output_path}")