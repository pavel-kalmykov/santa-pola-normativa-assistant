FROM python:3.13-slim

# opencv-python (a transitive docling dependency, pulled in for image
# preprocessing) needs these X11/GL runtime libraries even in a headless
# container; python:3.13-slim ships none of them by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies installed in their own layer first, so an app-code-only
# change doesn't force a full re-install of the embedding model deps.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra ingestion --no-install-project

COPY . .
RUN uv sync --frozen --extra ingestion

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "src/santa_pola_rag/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.headless=true"]
