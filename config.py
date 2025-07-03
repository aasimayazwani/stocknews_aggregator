import os

# Load from environment variables or hard-code for testing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

DEFAULT_MODEL = "gpt-4o"  # Can fallback to "gpt-3.5-turbo"
