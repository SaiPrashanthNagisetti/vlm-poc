import fitz
import pdfplumber
import os
import json


def detect_and_extract(pdf_path: str,
                       metadata_output_root="pdf_metadata_output",
                       asset_output_root="extracted_assets"):

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Create output folders
    os.makedirs(metadata_output_root, exist_ok=True)
    asset_folder = os.path.join(asset_output_root, pdf_name)
    os.makedirs(asset_folder, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    text_page_count = 0
    image_page_count = 0
    table_page_count = 0

    page_results = []

    with pdfplumber.open(pdf_path) as plumber_pdf:

        for page_number in range(total_pages):

            page = doc[page_number]
            plumber_page = plumber_pdf.pages[page_number]

            page_index = page_number + 1

            # ----------------------------------
            # TEXT
            # ----------------------------------
            text = page.get_text().strip()
            has_text = len(text.split()) > 10

            if has_text:
                text_page_count += 1

            # ----------------------------------
            # RASTER IMAGES
            # ----------------------------------
            image_list = page.get_images(full=True)
            image_count = len(image_list)
            has_images = image_count > 0

            if has_images:
                image_page_count += 1

                img_counter = 1
                for img in image_list:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    file_name = f"page_{page_index}_img_{img_counter}.{image_ext}"
                    file_path = os.path.join(asset_folder, file_name)

                    with open(file_path, "wb") as f:
                        f.write(image_bytes)

                    img_counter += 1

            # ----------------------------------
            # VECTOR GRAPHICS
            # ----------------------------------
            drawings = page.get_drawings()

            if drawings:
                pix = page.get_pixmap(dpi=200)

                file_name = f"page_{page_index}_vector_1.png"
                file_path = os.path.join(asset_folder, file_name)
                pix.save(file_path)

            # ----------------------------------
            # TABLES
            # ----------------------------------
            tables = plumber_page.find_tables()
            table_count = len(tables)
            has_tables = table_count > 0

            if has_tables:
                table_page_count += 1

                table_counter = 1
                for table in tables:
                    bbox = table.bbox
                    rect = fitz.Rect(bbox)

                    pix = page.get_pixmap(clip=rect, dpi=200)

                    file_name = f"page_{page_index}_table_{table_counter}.png"
                    file_path = os.path.join(asset_folder, file_name)
                    pix.save(file_path)

                    table_counter += 1

            page_results.append({
                "page_number": page_index,
                "has_text": has_text,
                "has_images": has_images,
                "has_tables": has_tables,
                "image_count": image_count,
                "table_count": table_count,
                "vector_graphics": len(drawings) > 0,
                "requires_image_processing": has_images or len(drawings) > 0
            })

    metadata = {
        "pdf_file_name": pdf_name,
        "total_pages": total_pages,
        "text_page_count": text_page_count,
        "image_page_count": image_page_count,
        "table_page_count": table_page_count,
        "page_results": page_results
    }

    metadata_output_path = os.path.join(
        metadata_output_root,
        f"{pdf_name}_metadata.json"
    )

    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Finished processing {pdf_name}")