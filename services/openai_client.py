from config.settings import OPENAI_API_KEY, API_URL, MODEL_NAME
import requests
import json

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
        "temperature": 0,
        "max_tokens": 2000
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()

    data = response.json()


    try:    
       response_text = data["choices"][0]["message"]["content"]
       structured_output = json.loads(response_text)
    except json.JSONDecodeError as e:
        print("⚠️ Invalid JSON returned by model")
        print(response_text)
        raise e

    usage = data.get("usage", {})

    return structured_output, usage