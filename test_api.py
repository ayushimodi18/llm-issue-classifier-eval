from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
if key:
    print(f"OPENAI_API_KEY loaded: {key[:5]}...{key[-4:]}")
else:
    print("OPENAI_API_KEY not set — copy .env.example to .env and add your key.")
