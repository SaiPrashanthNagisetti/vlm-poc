import fitz
import pdfplumber
import os

MIN_DRAWINGS_THRESHOLD = 5

TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 5,
    "join_tolerance": 5,
    "min_words_vertical": 2,
    "min_words_horizontal": 2,
}

def detect_pdf_content(pdf_path: str):
    doc      = fitz.open(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    total_pages = len(doc)

    text_page_count   = 0
    image_page_count  = 0
    table_page_count  = 0
    vector_page_count = 0

    page_results = []

    with pdfplumber.open(pdf_path) as plumber_pdf:
        for page_number in range(total_pages):
            page         = doc[page_number]
            plumber_page = plumber_pdf.pages[page_number]

            # TEXT
            text     = page.get_text().strip()
            has_text = len(text) > 50

            # RASTER IMAGES
            image_list  = page.get_images(full=True)
            image_count = len(image_list)
            has_images  = image_count > 0

            # VECTOR GRAPHICS
            drawings            = page.get_drawings()
            has_vector_graphics = len(drawings) > MIN_DRAWINGS_THRESHOLD

            # TABLES (text strategy handles both gridded and gridless)
            tables      = plumber_page.find_tables(TABLE_SETTINGS)
            table_count = len(tables)
            has_tables  = table_count > 0

            # Update counters
            if has_text:            text_page_count   += 1
            if has_images:          image_page_count  += 1
            if has_tables:          table_page_count  += 1
            if has_vector_graphics: vector_page_count += 1

            content_types = []
            if has_text:            content_types.append("text")
            if has_images:          content_types.append("images")
            if has_tables:          content_types.append("tables")
            if has_vector_graphics: content_types.append("vector_graphics")

            page_results.append({
                "page_number":               page_number + 1,
                "has_text":                  has_text,
                "has_images":                has_images,
                "has_tables":                has_tables,
                "image_count":               image_count,
                "table_count":               table_count,
                "raster_images":             image_count,
                "vector_graphics":           has_vector_graphics,
                "content_types":             content_types,
                "requires_image_processing": has_images or has_vector_graphics or has_tables
            })

    doc.close()

    return {
        "pdf_file_name":     pdf_name,
        "total_pages":       total_pages,
        "text_page_count":   text_page_count,
        "image_page_count":  image_page_count,
        "table_page_count":  table_page_count,
        "vector_page_count": vector_page_count,
        "page_results":      page_results
    }


