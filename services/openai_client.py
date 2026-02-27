import requests
import json
from config.settings import OPENAI_API_KEY, API_URL, MODEL_NAME

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

def call_openai(prompt: str, image_base64: str):
    payload = {
        "model": MODEL_NAME,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 600
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()

    data = response.json()
    #return json.loads(data["choices"][0]["message"]["content"])
    structured_output = json.loads(
        data["choices"][0]["message"]["content"]
    )

    usage = data.get("usage", {})

    return structured_output, usage