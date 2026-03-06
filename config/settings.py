import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

API_URL = "https://api.openai.com/v1/chat/completions"

MODEL_NAME = "gpt-4o"

INPUT_FOLDER = "input/data"
EXTRACTED_ASSETS = "extracted_assets"
VISUAL_OUTPUT = "visual_output"
MERGED_OUTPUT = "merged_output"

VECTOR_DB_PATH = "vector_store"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
TOP_K = 8