import os
import json
from config.settings import EXTRACTED_ASSETS
from utils.file_utils import extract_page_number
from services.image_analyzer import analyze_image

from concurrent.futures import ThreadPoolExecutor, as_completed


OUTPUT_ROOT = "output"


def process_single_image(folder_name, image_file, input_folder_path, output_folder_path):

    image_path = os.path.join(input_folder_path, image_file)

    relative_image_path = os.path.relpath(image_path)


    page_number = extract_page_number(image_file)
    model_output, usage = analyze_image(image_path)

    final_metadata = {
        "file_name": folder_name,
        "page_number": page_number,
        "image_title": model_output.get("image_title"),
        "image_keywords": model_output.get("image_keywords"),
        "summary": model_output.get("summary"),
        "table_rows": model_output.get("table_rows", []),
        "confidence_score": model_output.get("confidence_score"),
        "image_citation": relative_image_path,
        "token_usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens")
            }

    }

    json_filename = os.path.splitext(image_file)[0] + ".json"
    json_output_path = os.path.join(output_folder_path, json_filename)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=4)

    print(f"Saved: {json_output_path}")


#def process_images():

    #max_workers = 5  # Start with 4–5

    #with ThreadPoolExecutor(max_workers=max_workers) as executor:

     #   futures = []

      #  for folder_name in os.listdir(INPUT_FOLDER):
#
 #           input_folder_path = os.path.join(INPUT_FOLDER, folder_name)

  #          if not os.path.isdir(input_folder_path):
   #             continue

    #        output_folder_path = os.path.join(OUTPUT_ROOT, folder_name)
     #       os.makedirs(output_folder_path, exist_ok=True)

      #      for image_file in os.listdir(input_folder_path):

       #         if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
        #            continue

         #       futures.append(
          #          executor.submit(
           #             process_single_image,
            #            folder_name,
             #           image_file,
              #          input_folder_path,
               #         output_folder_path
                #    )
                #)

        #for future in as_completed(futures):
         #   future.result()

def process_images():

    max_workers = 5

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = []

        # Loop through each PDF folder
        for folder_name in os.listdir(EXTRACTED_ASSETS):

            pdf_folder_path = os.path.join(EXTRACTED_ASSETS, folder_name)

            if not os.path.isdir(pdf_folder_path):
                continue

            # Loop through asset categories
            for asset_type in ["images", "tables", "vector_graphics"]:

                input_asset_path = os.path.join(pdf_folder_path, asset_type)

                if not os.path.exists(input_asset_path):
                    continue

                # Create matching output structure
                output_asset_path = os.path.join(
                    OUTPUT_ROOT,
                    folder_name,
                    asset_type
                )
                os.makedirs(output_asset_path, exist_ok=True)

                # Loop through image files
                for image_file in os.listdir(input_asset_path):

                    if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue

                    futures.append(
                        executor.submit(
                            process_single_image,
                            folder_name,
                            image_file,
                            input_asset_path,
                            output_asset_path
                        )
                    )

        for future in as_completed(futures):
            future.result()