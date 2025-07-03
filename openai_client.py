import openai
from config import OPENAI_API_KEY, DEFAULT_MODEL

openai.api_key = OPENAI_API_KEY

def ask_openai(system_prompt: str, user_prompt: str) -> str:
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = openai.ChatCompletion.create(
            model=DEFAULT_MODEL,
            messages=messages
        )

        return response.choices[0].message["content"]

    except Exception as e:
        return f"OpenAI Error: {e}"
