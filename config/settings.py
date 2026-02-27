import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

API_URL = "https://api.openai.com/v1/chat/completions"

INPUT_FOLDER = "C:\\Users\\nsaip\\Documents\\VLM_POC\\extracted_assets"
OUTPUT_FILE = "output/results.json"
MODEL_NAME = "gpt-4o-mini"