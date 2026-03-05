import os
import sys

#from config.settings import OUTPUT_FOLDER

# processors
from processors.pdf_extractor import extract_pdf_assets
from processors.image_processor import process_images

# pipelines
from pipelines.merged_json import run_merge
from pipelines.rag_pipeline import build_index, query_rag



def main():

    print("\n==== VLM Pipeline ====\n")
    #print("1. Extract assets from PDFs")
    #print("2. Analyze images (VLM)")
    #print("3. Merge text + visuals")
    #print("4. Build vector index")
    print("1. Query")
    print("2. Run FULL pipeline\n")

    choice = input("Select an option (1-2): ").strip()

    if choice == "1":
        question = input("Enter your question: ")
        query_rag(question)

    #elif choice == "2":
     #   process_images()

    #elif choice == "3":
     #   run_merge()

    #elif choice == "4":
     #   build_index()

    #elif choice == "5":
     #   question = input("Enter your question: ")
      #  query_rag(question)

    elif choice == "2":
        print("\nRunning full pipeline...\n")
        extract_pdf_assets()
        process_images()
        run_merge()
        build_index()

        question = input("\nEnter your question: ")
        query_rag(question)

    else:
        print("Invalid option")


if __name__ == "__main__":
    main()