import os
import json
from config.settings import OUTPUT_FILE
from processors.image_processor import process_images

def main():
    os.makedirs("output", exist_ok=True)

    print("Starting image processing...\n")

    results = process_images()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nProcessing complete.")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()