from openai import OpenAI
from config import OPENAI_API_KEY, DEFAULT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

def ask_openai(model: str, system_prompt: str, user_prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI Error: {e}"
