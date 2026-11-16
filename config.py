import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_URL = os.getenv(
    "BASE_URL",
    "https://api.openai.com/v1"
)

# -----------------------------
# Models — GPT series via any OpenAI-compatible inference API
# Pricing: per-token rates below (USD)
# -----------------------------

MODELS = {
    "gpt4o": {
        "model": "gpt-4o-mini",
        "output_file": "Data/model_predictions_gpt4omini.csv",
        "cost_per_1k_input_tokens":  0.000150,  # $0.150 per 1M input tokens
        "cost_per_1k_output_tokens": 0.000600,  # $0.600 per 1M output tokens
    },
    "gpt41": {
        "model": "gpt-4.1-mini",
        "output_file": "Data/model_predictions_gpt41mini.csv",
        "cost_per_1k_input_tokens":  0.000400,  # $0.400 per 1M input tokens
        "cost_per_1k_output_tokens": 0.001600,  # $1.600 per 1M output tokens
    },
    "gpt51": {
        "model": "gpt-5.1",
        "output_file": "Data/model_predictions_gpt51.csv",
        "cost_per_1k_input_tokens":  0.002000,  # $2.00 per 1M input tokens (frontier)
        "cost_per_1k_output_tokens": 0.008000,  # $8.00 per 1M output tokens
    },
}

# -----------------------------
# Concurrency — configurable without rebuilding container
# Override: set MAX_WORKERS=20 in environment
# -----------------------------
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))

# -----------------------------
# Source repository for issues (owner/repo)
# -----------------------------
GITHUB_REPO = os.getenv("GITHUB_REPO", "digitalocean/doctl")
