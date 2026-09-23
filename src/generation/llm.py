import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o-mini"

# Without an explicit cap, OpenRouter reserves budget assuming the model
# could generate up to its full max output (often 16k+ tokens), which can
# get rejected with a 402 even when the account has enough balance for what
# the answer will actually cost. Our answers are a few sentences with a
# citation -- 1024 tokens is comfortably more than any real response needs.
MAX_TOKENS = 1024


def _get_api_key():
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to a .env file at the "
            "project root: OPENROUTER_API_KEY=sk-or-..."
        )

    return api_key


def generate_answer_with_usage(prompt, model_name=MODEL_NAME, max_tokens=MAX_TOKENS):
    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {_get_api_key()}"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()

    return {
        "text": data["choices"][0]["message"]["content"],
        "usage": data.get("usage", {}),
    }


def generate_answer(prompt, model_name=MODEL_NAME, max_tokens=MAX_TOKENS):
    return generate_answer_with_usage(prompt, model_name, max_tokens)["text"]


def generate_answer_stream(prompt, model_name=MODEL_NAME, max_tokens=MAX_TOKENS):
    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {_get_api_key()}"},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
        },
        timeout=60,
        stream=True,
    )
    response.raise_for_status()

    for raw_line in response.iter_lines():
        if not raw_line:
            continue

        line = raw_line.decode("utf-8")

        if line.startswith(":"):
            continue

        if not line.startswith("data: "):
            continue

        payload = line[len("data: "):]

        if payload == "[DONE]":
            break

        chunk = json.loads(payload)
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content")

        if content:
            yield content
