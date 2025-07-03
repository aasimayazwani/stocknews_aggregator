import os
from dotenv import load_dotenv
load_dotenv()
# Load from environment variables or hard-code for testing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DEFAULT_MODEL = "gpt-4o"  # Can fallback to "gpt-3.5-turbo"




# === Environment ===

#os.environ['OPENAI_API_KEY'] = 
#os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")