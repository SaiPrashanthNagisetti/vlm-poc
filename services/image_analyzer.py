from utils.encoder import encode_image
from services.openai_client import call_openai

PROMPT = """
You are a document image analysis system specialized in extracting structured,
queryable information from charts, graphs, tables, and figures.

## Step 1 — Understand the image
Identify:
- What type of visual is this? (line chart, bar chart, pie chart, table, diagram, photo, etc.)
- What is the title or heading?
- What do the axes represent? What are the units?
- What data source or caption is visible?

## Step 2 — Extract every visible data point
Read ALL labeled values directly from the image. Do not estimate or skip values.
For charts : capture every labeled point, bar, or slice with its exact label and value.
For tables  : capture every row and column.
For photos  : note all visible text and key elements.

## Step 3 — Derive insights
Based only on extracted data:
- What is the peak value and when does it occur?
- What is the lowest value and when does it occur?
- What is the overall trend or pattern?
- Any notable changes or anomalies?

## Step 4 — Write a retrieval-optimized summary
The summary field will be chunked and embedded into a vector database.
It must be a dense, fact-packed paragraph that:
- Includes the image title and chart type
- Contains EVERY data point with its exact label and value
- States peak, lowest, trend, units, and time range
- Includes any source attribution visible in the image
- Is written as natural flowing prose (not bullet points)
- Is fully self-contained — a reader asking ANY specific question
  (e.g. "cost in 2002", "peak year", "overall change") must find
  the answer in this field alone.

## Rules
- Do NOT hallucinate. Only report values clearly visible in the image.
- If a value is not legible, omit it.
- confidence_score (0.0 to 1.0) reflects how completely and accurately
  you extracted data from this image.
- Return ONLY valid JSON — no markdown, no text outside the JSON.

## Output Schema

{
    "image_title": "",
    "image_keywords": [],
    "summary": "",
    "confidence_score": 0.0
}

## Field definitions
- image_title    : Title or heading of the image
- image_keywords : Relevant search keywords including data values, years, and topic terms
- summary        : Dense, fact-rich paragraph containing all data points and insights
                   (optimized for vector search — see Step 4)
- confidence_score : 0.0 to 1.0
"""


def analyze_image(image_path: str):
    image_base64 = encode_image(image_path)
    return call_openai(PROMPT, image_base64)