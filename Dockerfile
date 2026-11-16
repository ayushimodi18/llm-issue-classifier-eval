FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Streamlit default port
EXPOSE 8501

# MAX_WORKERS and API keys injected at runtime via -e flags
ENV OPENAI_API_KEY=""
ENV BASE_URL="https://api.openai.com/v1"
ENV MAX_WORKERS=10

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
