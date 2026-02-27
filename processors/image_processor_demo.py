import os
from config.settings import INPUT_FOLDER
from utils.file_utils import extract_page_number
from services.image_analyzer import analyze_image

def process_images():
    results = []

    for folder_name in os.listdir(INPUT_FOLDER):
        folder_path = os.path.join(INPUT_FOLDER, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for image_file in os.listdir(folder_path):
            if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(folder_path, image_file)

            print(f"Processing: {image_path}")

            page_number = extract_page_number(image_file)
            model_output = analyze_image(image_path)

            results.append({
                "file_name": folder_name,
                "page_number": page_number,
                "image_title": model_output.get("image_title"),
                "image_keywords": model_output.get("image_keywords"),
                "summary": model_output.get("summary")
            })

    return results