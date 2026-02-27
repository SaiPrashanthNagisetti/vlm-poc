"""
PDF Content Extractor
=====================
Extracts tables, images, and vector graphics from PDFs as PNG files.

Detection  : pdfplumber  — finds what and where
Rendering  : PyMuPDF     — renders high-quality PNG crops

Install:
    pip install pymupdf pdfplumber

Output structure:
    output_folder/
    └── pdf_name/
        ├── tables/
        ├── images/
        └── vector_graphics/
"""

import os
import json
import fitz          # PyMuPDF  (pip install pymupdf)
import pdfplumber

from sklearn.feature_extraction.text import TfidfVectorizer
import re

# ── Settings ──────────────────────────────────────────────────────────────────

DPI            = 150    # render resolution
TABLE_PADDING  = 5      # pt padding around table crops
MIN_IMAGE_SIZE = 60     # ignore images smaller than this (pt)
MIN_RECT_COUNT = 20     # min rects on page to count as vector graphic
MIN_LINE_COUNT = 20     # min lines on page to count as vector graphic

HEADER_RATIO = 0.08   # top 8% of page removed
FOOTER_RATIO = 0.08   # bottom 8% removed

# ── Rendering (PyMuPDF) ───────────────────────────────────────────────────────

def crop_and_save(fitz_page, bbox, output_path, padding=0):
    """Crop a region from a PDF page and save as PNG."""
    x0, top, x1, bottom = bbox
    pw = fitz_page.rect.width
    ph = fitz_page.rect.height

    # Apply padding, clamped to page bounds
    x0     = max(x0     - padding, 0)
    top    = max(top    - padding, 0)
    x1     = min(x1     + padding, pw)
    bottom = min(bottom + padding, ph)

    scale  = DPI / 72
    matrix = fitz.Matrix(scale, scale)
    clip   = fitz.Rect(x0, top, x1, bottom)
    pix    = fitz_page.get_pixmap(matrix=matrix, clip=clip)
    pix.save(output_path)


def save_full_page(fitz_page, output_path):
    """Render and save an entire page as PNG."""
    matrix = fitz.Matrix(DPI / 72, DPI / 72)
    pix    = fitz_page.get_pixmap(matrix=matrix)
    pix.save(output_path)


# ── Table Detection (pdfplumber) ──────────────────────────────────────────────

def is_valid_table(table, page_w, page_h, plumber_page):
    """
    Returns True if the detected region looks like a real data table.
    Rejects: chart overlaps, full-page blobs, sparse regions, prose fragments.
    """
    data = table.extract()
    if not data:
        return False

    rows      = len(data)
    cols      = max(len(r) for r in data)
    non_empty = sum(1 for r in data for c in r if c and str(c).strip())

    # Too small
    if rows < 3 or cols < 2 or non_empty < 6:
        return False

    x0, top, x1, bottom = table.bbox
    h = bottom - top
    w = x1 - x0

    # Row height sanity — huge = chart bbox merged with table
    if h / rows > 40:
        return False

    # Covers too much of the page — whole-page false positive
    if (w * h) / (page_w * page_h) > 0.6:
        return False

    # Too sparse
    if non_empty / (rows * cols) < 0.15:
        return False

    # Prose fragment check — inline PDF lines can chop paragraphs into fake tables
    cells = [str(c).strip() for r in data for c in r if c and str(c).strip()]
    connector_words = {'to', 'by', 'primarily', 'driven', 'due',
                       'reflecting', 'compared', 'resulting', 'offset'}
    if sum(1 for c in cells if c.lower().strip('., ') in connector_words) >= 3:
        return False

    # High word density in thin bbox = paragraph, not a table
    try:
        region     = plumber_page.crop((x0, top, x1, bottom))
        word_count = len(region.extract_words())
        if w * h > 0 and (word_count / (w * h) * 1000) > 3.8 and h < 80:
            return False
    except Exception:
        pass

    return True


def merge_nearby_images(images, gap=20):
    """
    Merges image bboxes that are close or overlapping into single bboxes.
    PDFs often store one visual image as multiple separate image objects
    (e.g. a chart split into left/right/bottom tiles).

    Args:
        images : list of pdfplumber image dicts
        gap    : max pt distance between images to still merge them

    Returns:
        list of merged (x0, top, x1, bottom) tuples
    """
    if not images:
        return []

    bboxes = [(img["x0"], img["top"], img["x1"], img["bottom"]) for img in images]

    # Repeatedly merge until no more merges happen
    changed = True
    while changed:
        changed = False
        result  = []
        used    = [False] * len(bboxes)

        for i, a in enumerate(bboxes):
            if used[i]:
                continue
            ax0, at, ax1, ab = a
            for j, b in enumerate(bboxes):
                if i == j or used[j]:
                    continue
                bx0, bt, bx1, bb = b
                # Merge if bboxes overlap or are within gap pt of each other
                h_close = ax0 - gap <= bx1 and bx0 - gap <= ax1
                v_close = at  - gap <= bb  and bt  - gap <= ab
                if h_close and v_close:
                    ax0, at  = min(ax0, bx0), min(at, bt)
                    ax1, ab  = max(ax1, bx1), max(ab, bb)
                    used[j]  = True
                    changed  = True
            result.append((ax0, at, ax1, ab))
            used[i] = True

        bboxes = result

    return bboxes


def get_rect_x_positions(plumber_page):
    """
    Returns x-positions of page rects taller than a text line.
    Used as explicit vertical column boundaries for rect-defined tables.
    Thin rects (<=10pt) are text highlights, not column borders — skip them.
    """
    positions = set()
    for r in plumber_page.rects:
        if (r['bottom'] - r['top']) > 10:
            positions.add(round(r['x0'], 1))
            positions.add(round(r['x1'], 1))
    return sorted(positions)


def detect_tables(plumber_page):
    """
    Finds tables on a page using two strategies:
      1. Lines    — standard gridded tables (primary)
      2. Explicit — tables whose columns are defined by filled rects,
                    not lines (e.g. shaded column headers).
                    Only replaces line results when it consolidates fragments.
    Returns a deduplicated list of valid tables.
    """
    page_w = plumber_page.width
    page_h = plumber_page.height

    def bbox_overlap_ratio(a, b):
        """Fraction of bbox a that is covered by bbox b."""
        ax0, at, ax1, ab = a
        bx0, bt, bx1, bb = b
        ix = max(0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0, min(ab,  bb)  - max(at,  bt))
        area_a = (ax1 - ax0) * (ab - at)
        return (ix * iy / area_a) if area_a > 0 else 0

    # ── Strategy 1: line-based ────────────────────────────────────────────────
    line_tables = []
    try:
        for t in plumber_page.find_tables({
            "vertical_strategy":   "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 5,
            "join_tolerance":  5,
        }):
            if is_valid_table(t, page_w, page_h, plumber_page):
                line_tables.append(t)
    except Exception:
        pass

    # ── Strategy 2: explicit vertical columns ─────────────────────────────────
    # When the line strategy fragments a single table (e.g. shaded header tables),
    # the explicit strategy finds it as one bbox. Replace the fragments with it.
    x_positions = get_rect_x_positions(plumber_page)
    if len(x_positions) >= 3:
        try:
            for exp_t in plumber_page.find_tables({
                "vertical_strategy":       "explicit",
                "horizontal_strategy":     "lines",
                "explicit_vertical_lines": x_positions,
                "snap_tolerance": 5,
                "join_tolerance":  5,
            }):
                if not is_valid_table(exp_t, page_w, page_h, plumber_page):
                    continue

                # Which line tables does this explicit table overlap?
                fragments = [
                    t for t in line_tables
                    if bbox_overlap_ratio(t.bbox, exp_t.bbox) > 0.4
                ]

                # Only consolidate when covering 2+ fragments at similar total area
                if len(fragments) >= 2:
                    frag_area = sum((t.bbox[2]-t.bbox[0]) * (t.bbox[3]-t.bbox[1])
                                    for t in fragments)
                    exp_area  = ((exp_t.bbox[2] - exp_t.bbox[0]) *
                                 (exp_t.bbox[3] - exp_t.bbox[1]))
                    if 0.7 <= exp_area / frag_area <= 1.5:
                        for f in fragments:
                            line_tables.remove(f)
                        line_tables.append(exp_t)
        except Exception:
            pass

    # ── Remove nested tables (keep outermost only) ────────────────────────────
    final = []
    for i, t in enumerate(line_tables):
        ax0, at, ax1, ab = t.bbox
        is_nested = any(
            i != j
            and ax0 >= line_tables[j].bbox[0]
            and at  >= line_tables[j].bbox[1]
            and ax1 <= line_tables[j].bbox[2]
            and ab  <= line_tables[j].bbox[3]
            for j in range(len(line_tables))
        )
        if not is_nested:
            final.append(t)

    return final


def expand_to_include_labels(plumber_page, bbox):
    """
    Expands a table bbox leftward to capture row label text that sits in a
    separate column just to the left of the data columns.
    Only expands when a dense band of label words is found nearby.
    """
    x0, top, x1, bottom = bbox

    # Skip if table already starts near the left margin
    if x0 < 200:
        return bbox

    table_h    = bottom - top
    candidates = [
        w for w in plumber_page.extract_words()
        if w['x0'] < x0 - 5           # left of data columns
        and w['x0'] > x0 - 220        # not too far left
        and w['top']    >= top - 5
        and w['bottom'] <= bottom + 5
    ]

    if not candidates:
        return bbox

    # Words must span at least 40% of the table height (rules out stray words)
    y_spread = max(w['top'] for w in candidates) - min(w['top'] for w in candidates)
    if y_spread < table_h * 0.4:
        return bbox

    # Label column must be close to the data (gap <= 80pt)
    if x0 - max(w['x1'] for w in candidates) > 80:
        return bbox

    return (min(w['x0'] for w in candidates) - 2, top, x1, bottom)


# ── Clean Text Extraction ─────────────────────────────────────────────────

#def extract_text_excluding_tables(plumber_page, table_bboxes):
    words = plumber_page.extract_words()
    page_height = plumber_page.height

    header_cutoff = page_height * HEADER_RATIO
    footer_cutoff = page_height * (1 - FOOTER_RATIO)

    filtered_words = []

    for w in words:
        wx0, wy0, wx1, wy1 = w["x0"], w["top"], w["x1"], w["bottom"]

        # Remove header/footer
        if wy1 < header_cutoff or wy0 > footer_cutoff:
            continue

        # Remove table region text
        inside_table = False
        for bx0, btop, bx1, bbottom in table_bboxes:
            if not (wx1 < bx0 or wx0 > bx1 or wy1 < btop or wy0 > bbottom):
                inside_table = True
                break

        if not inside_table:
            filtered_words.append(w)

    # Reconstruct text by line
    lines = {}
    for w in filtered_words:
        line_key = round(w["top"], 1)
        lines.setdefault(line_key, []).append(w)

    final_lines = []
    for _, line_words in sorted(lines.items()):
        sorted_words = sorted(line_words, key=lambda x: x["x0"])
        final_lines.append(" ".join(w["text"] for w in sorted_words))

    return "\n".join(final_lines).strip()

def extract_text_excluding_tables_and_charts(plumber_page, table_bboxes):
    words = plumber_page.extract_words()
    page_height = plumber_page.height

    header_cutoff = page_height * HEADER_RATIO
    footer_cutoff = page_height * (1 - FOOTER_RATIO)

    # Detect vector-heavy page
    has_vector = (
        len(plumber_page.rects) >= MIN_RECT_COUNT or
        len(plumber_page.lines) >= MIN_LINE_COUNT
    )

    # If vector heavy, assume bottom 45% contains chart
    chart_cutoff = page_height * 0.55 if has_vector else page_height

    filtered_words = []

    for w in words:
        wx0, wy0, wx1, wy1 = w["x0"], w["top"], w["x1"], w["bottom"]

        # Remove header/footer
        if wy1 < header_cutoff or wy0 > footer_cutoff:
            continue

        # Remove chart region text (bottom area)
        if has_vector and wy0 > chart_cutoff:
            continue

        # Remove table region text
        inside_table = False
        for bx0, btop, bx1, bbottom in table_bboxes:
            if not (wx1 < bx0 or wx0 > bx1 or wy1 < btop or wy0 > bbottom):
                inside_table = True
                break

        if not inside_table:
            filtered_words.append(w)

    # Reconstruct lines
    lines = {}
    for w in filtered_words:
        line_key = round(w["top"], 1)
        lines.setdefault(line_key, []).append(w)

    final_lines = []
    for _, line_words in sorted(lines.items()):
        sorted_words = sorted(line_words, key=lambda x: x["x0"])
        final_lines.append(" ".join(w["text"] for w in sorted_words))

    return "\n".join(final_lines).strip()


def extract_keywords_tfidf(text, top_k=12):
    """
    Extract unique keywords using TF-IDF from a single page text.
    """

    # Basic cleanup
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    if len(text.split()) < 5:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=2000,
            ngram_range=(1, 2)  # unigrams + bigrams (better than only single words)
        )

        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        sorted_indices = scores.argsort()[::-1]

        keywords = []
        for idx in sorted_indices:
            if scores[idx] <= 0:
                continue
            keywords.append(feature_names[idx])
            if len(keywords) >= top_k:
                break

        return list(set(keywords))  # ensure uniqueness

    except Exception:
        return []

# ── Main Extraction ────────────────────────────────────────────────────────────

def extract_pdf(pdf_path, output_folder="extracted_assets", dpi=DPI):

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir  = os.path.join(output_folder, pdf_name)

    tables_dir  = os.path.join(out_dir, "tables")
    images_dir  = os.path.join(out_dir, "images")
    vectors_dir = os.path.join(out_dir, "vector_graphics")
    text_dir    = os.path.join(out_dir, "text")

    for d in [tables_dir, images_dir, vectors_dir, text_dir]:
        os.makedirs(d, exist_ok=True)

    fitz_doc = fitz.open(pdf_path)
    results  = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"\nProcessing: {pdf_name}  ({total} pages)\n")

        for i, plumber_page in enumerate(pdf.pages):
            fitz_page = fitz_doc[i]
            page_num  = i + 1
            assets    = {"tables": [], "images": [], "vectors": []}

            # ───────────────────────────────────────────────
            # TABLE DETECTION
            # ───────────────────────────────────────────────
            table_objects = detect_tables(plumber_page)
            table_bboxes  = [t.bbox for t in table_objects]

            # Calculate table coverage ratio
            page_area = plumber_page.width * plumber_page.height
            table_area = sum(
                (bx1 - bx0) * (bbottom - btop)
                for bx0, btop, bx1, bbottom in table_bboxes
            )
            table_coverage = table_area / page_area if page_area > 0 else 0

            # ───────────────────────────────────────────────
            # SAVE TABLE IMAGES
            # ───────────────────────────────────────────────
            for n, table in enumerate(table_objects, 1):
                fname = f"page{page_num}_table{n}.png"
                path  = os.path.join(tables_dir, fname)
                bbox  = expand_to_include_labels(plumber_page, table.bbox)
                try:
                    crop_and_save(fitz_page, bbox, path, padding=TABLE_PADDING)
                    assets["tables"].append(path)
                    print(f"  [TABLE]  p{page_num} -> {fname}")
                except Exception as e:
                    print(f"  [ERROR]  table p{page_num}: {e}")

            # ───────────────────────────────────────────────
            # IMAGE EXTRACTION
            # ───────────────────────────────────────────────
            raw_images = [
                img for img in plumber_page.images
                if (img["x1"] - img["x0"]) >= MIN_IMAGE_SIZE
                and (img["bottom"] - img["top"]) >= MIN_IMAGE_SIZE
            ]

            for n, bbox in enumerate(merge_nearby_images(raw_images), 1):
                fname = f"page{page_num}_img{n}.png"
                path  = os.path.join(images_dir, fname)
                try:
                    crop_and_save(fitz_page, bbox, path)
                    assets["images"].append(path)
                    print(f"  [IMAGE]  p{page_num} -> {fname}")
                except Exception as e:
                    print(f"  [ERROR]  image p{page_num}: {e}")

            # ───────────────────────────────────────────────
            # VECTOR GRAPHICS
            # ───────────────────────────────────────────────
            has_vector = (
                len(plumber_page.rects)  >= MIN_RECT_COUNT or
                len(plumber_page.lines) >= MIN_LINE_COUNT
            )

            if has_vector:
                fname = f"page{page_num}_vector.png"
                path  = os.path.join(vectors_dir, fname)
                try:
                    save_full_page(fitz_page, path)
                    assets["vectors"].append(path)
                    print(f"  [VECTOR] p{page_num} -> {fname}")
                except Exception as e:
                    print(f"  [ERROR]  vector p{page_num}: {e}")

            # ───────────────────────────────────────────────
            # CLEAN TEXT EXTRACTION (NEW LOGIC)
            # ───────────────────────────────────────────────

            # 🚀 If page is mostly table, skip text completely
            if table_coverage > 0.7:
                clean_text = ""
                print(f"  [TEXT]   p{page_num} skipped (table-heavy page)")
            else:
                clean_text = extract_text_excluding_tables_and_charts(
                    plumber_page,
                    table_bboxes
                )

            if clean_text:

                keywords = extract_keywords_tfidf(clean_text, top_k=12)
                json_path = os.path.join(text_dir, f"pgno_{page_num}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "pdf": pdf_name,
                        "page": page_num,
                        "text": clean_text,
                        "keywords": keywords
                    }, f, indent=2, ensure_ascii=False)

            results.append({
                "page": page_num,
                "text_length": len(clean_text),
                "tables": len(assets["tables"]),
                "images": len(assets["images"]),
                "vectors": len(assets["vectors"]),
                "files": assets
            })

    fitz_doc.close()

    print("\nExtraction Complete")
    print(f"Output → {out_dir}\n")

    return results

# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    INPUT_FOLDER  = "input/data"
    OUTPUT_FOLDER = "extracted_assets"

    # Find all PDFs in the input folder
    pdf_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in: {INPUT_FOLDER}")
    else:
        print(f"Found {len(pdf_files)} PDF(s) in {INPUT_FOLDER}\n")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(INPUT_FOLDER, pdf_file)
            print(f"{'='*50}")
            print(f"Processing: {pdf_file}")
            print(f"{'='*50}")
            try:
                extract_pdf(
                    pdf_path=pdf_path,
                    output_folder=OUTPUT_FOLDER,
                    dpi=150,
                )
            except Exception as e:
                print(f"[ERROR] Failed to process {pdf_file}: {e}")




