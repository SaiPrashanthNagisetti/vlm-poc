from utils.encoder import encode_image
from services.openai_client import call_openai

PROMPT = """
You are a document image analysis system specialized in extracting structured,
queryable information from any type of visual content found in documents.

## Step 1 - Identify the visual type
First determine what kind of visual this is:

- LINE CHART      : data points connected by lines over an axis
- BAR CHART       : vertical or horizontal bars comparing values
- PIE CHART       : circular chart showing proportions
- TABLE           : rows and columns of data
- FLOW DIAGRAM    : boxes/shapes connected by arrows showing a process or workflow
- ORG CHART       : hierarchy of roles or entities
- MAP             : geographic or spatial diagram
- PHOTO           : real-world photograph
- DIAGRAM         : technical or conceptual illustration
- OTHER           : anything that does not fit above

## Step 2 - Extract content based on visual type

For CHARTS (line, bar, pie):
  - Capture every labeled data point with its exact label and value
  - Note axis labels, units, legend entries, and data source

For TABLES:
  - Capture every row and column as structured data
  - Note headers, subheaders, and any footnotes

For FLOW DIAGRAMS / ORG CHARTS:
  - List every node/box and its label
  - Describe the connections and direction of arrows
  - Identify the start point, end point, and any decision branches
  - Describe the overall process or hierarchy from top to bottom

For PHOTOS:
  - Describe the scene, people, objects, and any visible text

For MAPS / DIAGRAMS:
  - Describe all labeled regions, components, and relationships

## Step 3 - Derive insights

For CHARTS       : peak, lowest, trend, anomalies
For TABLES       : key comparisons, highest/lowest rows, patterns
For FLOW DIAGRAMS: what process is shown, how many steps, any branches or loops,
                   what triggers the process and what is the outcome
For PHOTOS       : what is happening, what is the context
For DIAGRAMS/MAPS: what system or concept is being illustrated

## Step 4 - Write a retrieval-optimized summary
The summary field will be chunked and embedded into a vector database.
It must be a dense, fact-packed paragraph that:
- States the visual type and title upfront
- For charts    : contains EVERY data point with label and value, peak, lowest, trend, units
- For tables    : contains all key values and comparisons
- For flows     : describes every step by name, arrow direction, decision points,
                  branches, start-to-end path, and the overall purpose of the process
- For photos    : describes the scene with enough detail to answer questions about it
- Is written as natural flowing prose (not bullet points)
- Is fully self-contained — a reader asking ANY question about this visual
  must find the answer in this field alone
- Includes any source, caption, or attribution visible in the image

## Rules
- Do NOT hallucinate. Only report what is clearly visible in the image.
- If a value or label is not legible, omit it.
- confidence_score (0.0 to 1.0) reflects how completely and accurately
  you extracted content from this image.
- Return ONLY valid JSON — no markdown, no text outside the JSON.

## Output Schema

{
    "image_title": "",
    "image_keywords": [],
    "summary": "",
    "confidence_score": 0.0
}

## Field definitions
- image_title    : Title or heading of the image; if none is visible, write a short
                   descriptive title based on the content (e.g. "Employee Onboarding Flow")
- image_keywords : Relevant search keywords — include node labels, step names,
                   data values, years, process names, and topic terms
- summary        : Dense, fact-rich paragraph optimized for vector search (see Step 4)
- confidence_score : 0.0 to 1.0
"""


def analyze_image(image_path: str):
    image_base64 = encode_image(image_path)
    return call_openai(PROMPT, image_base64)