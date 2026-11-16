# LLM GitHub Issue Classification — Evaluation Dashboard

Evaluates GPT models on GitHub issue classification using any OpenAI-compatible inference API and recommends a production model based on accuracy, cost, and operational performance.

## Models Evaluated

| Key | Model | Input $/1K tokens | Output $/1K tokens |
|-----|-------|-------------------|---------------------|
| `gpt4o` | `gpt-4o-mini`  | $0.000150 | $0.000600 |
| `gpt41` | `gpt-4.1-mini` | $0.000400 | $0.001600 |
| `gpt51` | `gpt-5.1`      | $0.002000 | $0.008000 |

## Label Set

`bug` · `enhancement` · `question` · `documentation` · `security` · `other`

## API Key Setup

> **No API keys are included in this repository.** All keys shown below are placeholders.

1. Get an API key from your inference provider (e.g. [OpenAI](https://platform.openai.com/api-keys), or any OpenAI-compatible endpoint).
2. Copy the example env file and add your key:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and replace `sk-your-openai-api-key-here` with your real key.
4. Verify it loads (prints a masked key):
   ```bash
   python test_api.py
   ```

`.env` is listed in `.gitignore` and `.dockerignore`, so your real key is never committed or baked into the Docker image — pass it at runtime with `-e OPENAI_API_KEY=...` instead.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | API key for the OpenAI-compatible endpoint |
| `BASE_URL` | | `https://api.openai.com/v1` | Inference endpoint (any OpenAI-compatible URL) |
| `GITHUB_REPO` | | `digitalocean/doctl` | `owner/repo` to fetch issues from (`github_loader.py`) |
| `MAX_WORKERS` | | `10` | Parallel inference concurrency (no rebuild needed) |

Copy `.env.example` to `.env` for local runs (`.env` is git-ignored):
```
OPENAI_API_KEY=sk-your-openai-api-key-here
BASE_URL=https://api.openai.com/v1
MAX_WORKERS=10
```

## Installation

```bash
pip install -r requirements.txt
```

## Full Run — Step by Step

```bash
# 1. Fetch issues from GitHub
python github_loader.py

# 2. Build ground truth (labeled subset)
python build_ground_truth.py

# 3. Build unscored corpus
python build_unscored_corpus.py

# 4. Classify scored corpus — all 3 models
python classifier_parallel.py gpt4o
python classifier_parallel.py gpt41
python classifier_parallel.py gpt51

# 5. Classify unscored corpus
python classifier_unscored_parallel.py gpt4o
python classifier_unscored_parallel.py gpt41
python classifier_unscored_parallel.py gpt51

# 6. Build evaluation results
python build_model_results.py parallel

# 7. Build disagreements + predictions
python build_full_predictions.py
python build_disagreements.py
python build_unscored_predictions.py
python build_unscored_agreement.py
python build_unscored_distribution.py

# 8. Launch dashboard
streamlit run app.py
```

Open: http://localhost:8501

## Docker

### Build
```bash
docker build -t issue-classifier-eval .
```

### Run
```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-your-openai-api-key-here \
  -e BASE_URL=https://api.openai.com/v1 \
  -e MAX_WORKERS=10 \
  issue-classifier-eval
```

Change concurrency without rebuilding:
```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-your-openai-api-key-here -e MAX_WORKERS=20 issue-classifier-eval
```

Open: http://localhost:8501

## Live Application

> Add your deployed URL here.

## Key Design Decisions

Why per-issue inference ?
Per-call cost, per-call latency, individually-retryable failures, and per-issue fallback routing all require separate requests.

**Wall clock time measurement**
Real timer wrapping the full parallel execution — not sum of serial latencies (which overcounts by the concurrency factor). Throughput = `total_issues / wall_clock`.

Cost calculation (traceable)
`call_cost = (prompt_tokens / 1000 × input_rate) + (completion_tokens / 1000 × output_rate)`
Stored per row in output CSVs. Total = `df["call_cost"].sum()`. Visible in Cost Trace section of dashboard.

Concurrency tradeoffs
Higher concurrency reduces wall-clock time but increases p95 latency under rate limiting. Configurable via `MAX_WORKERS` env var — no container rebuild needed.

Ground truth methodology
Maps maintainer GitHub labels to the 6-class schema conservatively — only unambiguous labels included. Avoids polluting eval with noisy or multi-label issues.

Failure handling
- Rate limit (429): exponential backoff + retry → fallback model
- Timeout: single retry → human review queue
- `security` class: always human review
- Schema violation (label not in valid set): re-run on fallback
