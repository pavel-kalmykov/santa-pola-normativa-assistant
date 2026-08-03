FROM python:3.13-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies installed in their own layer first, so an app-code-only
# change doesn't force a full re-install of the embedding model deps.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/santa_pola_rag/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.headless=true"]
