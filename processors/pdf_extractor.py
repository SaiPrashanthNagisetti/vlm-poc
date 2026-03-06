"""
PDF Content Extractor
Extract tables, images, vector charts and clean text.

Detection  : pdfplumber
Rendering  : PyMuPDF
"""

import os
import json
import fitz
import pdfplumber
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from config.settings import INPUT_FOLDER, EXTRACTED_ASSETS


# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

DPI = 150
TABLE_PADDING = 5

MIN_IMAGE_SIZE = 60
MIN_RECT_COUNT = 20
MIN_LINE_COUNT = 20

HEADER_RATIO = 0.08
FOOTER_RATIO = 0.08


# ─────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────

def crop_and_save(fitz_page, bbox, output_path, padding=0):

    x0, top, x1, bottom = bbox

    pw = fitz_page.rect.width
    ph = fitz_page.rect.height

    x0 = max(x0 - padding, 0)
    top = max(top - padding, 0)

    x1 = min(x1 + padding, pw)
    bottom = min(bottom + padding, ph)

    scale = DPI / 72
    matrix = fitz.Matrix(scale, scale)

    clip = fitz.Rect(x0, top, x1, bottom)

    pix = fitz_page.get_pixmap(matrix=matrix, clip=clip)

    pix.save(output_path)


# ─────────────────────────────────────────────
# VECTOR CHART DETECTION
# ─────────────────────────────────────────────

def detect_vector_bbox(plumber_page):

    elements = plumber_page.rects + plumber_page.lines

    if len(elements) < 15:
        return None

    xs = [e["x0"] for e in elements]
    ys = [e["top"] for e in elements]
    xe = [e["x1"] for e in elements]
    ye = [e["bottom"] for e in elements]

    return (min(xs), min(ys), max(xe), max(ye))


# ─────────────────────────────────────────────
# IMAGE MERGING
# ─────────────────────────────────────────────

def merge_nearby_images(images, gap=20):

    if not images:
        return []

    bboxes = [(img["x0"], img["top"], img["x1"], img["bottom"]) for img in images]

    changed = True

    while changed:

        changed = False
        result = []
        used = [False] * len(bboxes)

        for i, a in enumerate(bboxes):

            if used[i]:
                continue

            ax0, at, ax1, ab = a

            for j, b in enumerate(bboxes):

                if i == j or used[j]:
                    continue

                bx0, bt, bx1, bb = b

                h_close = ax0 - gap <= bx1 and bx0 - gap <= ax1
                v_close = at - gap <= bb and bt - gap <= ab

                if h_close and v_close:

                    ax0 = min(ax0, bx0)
                    at = min(at, bt)

                    ax1 = max(ax1, bx1)
                    ab = max(ab, bb)

                    used[j] = True
                    changed = True

            result.append((ax0, at, ax1, ab))
            used[i] = True

        bboxes = result

    return bboxes


# ─────────────────────────────────────────────
# FIX BROKEN TEXT (important for vector PDFs)
# ─────────────────────────────────────────────

def fix_spaced_text(text):

    # Fix cases like: R e p o r t e d
    text = re.sub(r'(?<=\b[A-Za-z])\s(?=[A-Za-z]\b)', '', text)

    # Remove excessive spacing
    text = re.sub(r'\s{2,}', ' ', text)

    return text


# ─────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_text_excluding_regions(plumber_page, table_bboxes, vector_bbox):

    words = plumber_page.extract_words()

    page_height = plumber_page.height

    header_cutoff = page_height * HEADER_RATIO
    footer_cutoff = page_height * (1 - FOOTER_RATIO)

    filtered_words = []

    for w in words:

        wx0, wy0, wx1, wy1 = w["x0"], w["top"], w["x1"], w["bottom"]

        if wy1 < header_cutoff or wy0 > footer_cutoff:
            continue

        inside_table = False

        for bx0, btop, bx1, bbottom in table_bboxes:

            if not (wx1 < bx0 or wx0 > bx1 or wy1 < btop or wy0 > bbottom):
                inside_table = True
                break

        if inside_table:
            continue

        if vector_bbox:

            vx0, vtop, vx1, vbottom = vector_bbox

            if not (wx1 < vx0 or wx0 > vx1 or wy1 < vtop or wy0 > vbottom):
                continue

        filtered_words.append(w)

    lines = {}

    for w in filtered_words:

        key = round(w["top"], 1)
        lines.setdefault(key, []).append(w)

    final_lines = []

    for _, line_words in sorted(lines.items()):

        sorted_words = sorted(line_words, key=lambda x: x["x0"])

        final_lines.append(" ".join(w["text"] for w in sorted_words))

    text = "\n".join(final_lines).strip()

    return fix_spaced_text(text)


# ─────────────────────────────────────────────
# KEYWORD EXTRACTION
# ─────────────────────────────────────────────

def extract_keywords_tfidf(text, top_k=12):

    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    if len(text.split()) < 5:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=2000,
        ngram_range=(1, 2)
    )

    tfidf = vectorizer.fit_transform([text])

    scores = tfidf.toarray()[0]
    words = vectorizer.get_feature_names_out()

    idx = scores.argsort()[::-1]

    keywords = []

    for i in idx:

        if scores[i] <= 0:
            continue

        keywords.append(words[i])

        if len(keywords) >= top_k:
            break

    return list(set(keywords))


# ─────────────────────────────────────────────
# MAIN EXTRACTION
# ─────────────────────────────────────────────

def extract_pdf(pdf_path, output_folder):

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    out_dir = os.path.join(output_folder, pdf_name)

    tables_dir = os.path.join(out_dir, "tables")
    images_dir = os.path.join(out_dir, "images")
    vectors_dir = os.path.join(out_dir, "vector_graphics")
    text_dir = os.path.join(out_dir, "text")

    for d in [tables_dir, images_dir, vectors_dir, text_dir]:
        os.makedirs(d, exist_ok=True)

    fitz_doc = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:

        for i, plumber_page in enumerate(pdf.pages):

            page_num = i + 1
            fitz_page = fitz_doc[i]

            tables = plumber_page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            # ─────────────────────────
            # TABLES
            # ─────────────────────────

            for n, table in enumerate(tables, 1):

                path = os.path.join(tables_dir, f"page{page_num}_table{n}.png")

                crop_and_save(fitz_page, table.bbox, path, TABLE_PADDING)

            # ─────────────────────────
            # IMAGES
            # ─────────────────────────

            raw_images = [
                img for img in plumber_page.images
                if (img["x1"] - img["x0"]) >= MIN_IMAGE_SIZE
                and (img["bottom"] - img["top"]) >= MIN_IMAGE_SIZE
            ]

            merged = merge_nearby_images(raw_images)

            for n, bbox in enumerate(merged, 1):

                path = os.path.join(images_dir, f"page{page_num}_img{n}.png")

                crop_and_save(fitz_page, bbox, path)

            # ─────────────────────────
            # VECTOR GRAPHICS
            # ─────────────────────────

            vector_bbox = detect_vector_bbox(plumber_page)

            if vector_bbox:

                path = os.path.join(vectors_dir, f"page{page_num}_vector.png")

                crop_and_save(fitz_page, vector_bbox, path)

            # ─────────────────────────
            # CLEAN TEXT
            # ─────────────────────────

            if vector_bbox and len(plumber_page.rects) > 50:
                clean_text = ""
            else:
                clean_text = extract_text_excluding_regions(
                    plumber_page,
                    table_bboxes,
                    vector_bbox
                )

            if clean_text:

                keywords = extract_keywords_tfidf(clean_text)

                json_path = os.path.join(text_dir, f"pgno_{page_num}.json")

                with open(json_path, "w", encoding="utf-8") as f:

                    json.dump({
                        "pdf": pdf_name,
                        "page": page_num,
                        "text": clean_text,
                        "keywords": keywords
                    }, f, indent=2)

    fitz_doc.close()

    print(f"Extraction complete → {out_dir}")


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────

def extract_pdf_assets():

    pdf_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    for pdf_file in pdf_files:

        pdf_path = os.path.join(INPUT_FOLDER, pdf_file)

        print(f"\nProcessing {pdf_file}\n")

        extract_pdf(pdf_path, EXTRACTED_ASSETS)