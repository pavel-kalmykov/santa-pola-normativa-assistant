# Santa Pola Municipal Ordinances Assistant

This is a multilingual, conversational RAG assistant over the public municipal ordinances, tax ordinances, bylaws and public notices ("bandos") of Santa Pola, Spain, built as the capstone project for the [LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp) course.

Residents of Santa Pola come from dozens of countries, and the official source documents are only published in Spanish. This assistant lets anyone ask "How much is the dog census fee?" or "Quand puis-je installer une terrasse sur la voie publique ?" in their own language and get an answer grounded in, and cited from, the actual ordinance.

## Problem

Santa Pola's town hall publishes ~1,470 PDFs across 22 sections of [santapola.es/ayuntamiento](https://santapola.es/ayuntamiento/). Finding the right fee, deadline or rule means knowing which of dozens of PDFs to open, in Spanish, often in a scanned/non-searchable format. This project builds a real ingestion pipeline over four "hard regulation" categories (268 PDFs), the ones residents and small businesses actually need to cite, and a conversational assistant that always answers with a source citation (document, page, URL), never from memory alone.

## Dataset

Scraped live from santapola.es, not the course FAQ:

| Category | Slug | PDFs |
|---|---|---|
| Ordenanzas Fiscales (tax ordinances) | `ordenanzas-fiscales` | 182 |
| Reglamentos y otras Ordenanzas (bylaws) | `reglamentos-otras-ordenanzas` | 53 |
| Bandos (public notices) | `bandos` | 23 |
| Normativas (regulations) | `normativas` | 10 |
| **Total** | | **268** |

268 PDFs produce 2,804 pages (352 of them scanned/image-only and OCR'd) and 12,089 indexed chunks. The category list (`santa_pola_rag/ingestion/scraper.py`) is a plain dict of slug to name, so extending to any of the other 18 sections on the town hall's site is a one-line change.

## Architecture

```mermaid
flowchart TD
    A[santapola.es] -->|scrape + download| B[("MinIO<br/>raw PDF bytes")]
    B --> C[PyMuPDF text extraction]
    C -->|text layer usable| F
    C -->|scanned or garbled| D["Vision OCR<br/>Gemini 2.5 Flash"]
    D --> F[dlt pipeline<br/>idempotent, merge writes]
    F --> G[("Postgres<br/>staging: metadata + text")]
    G --> H[Chunking]
    H --> I[Multilingual embeddings]
    I --> J[("Qdrant<br/>vectors")]
    I --> K[("Elasticsearch<br/>BM25 chunks")]
    J --> L["Hybrid search<br/>RRF fusion + cross-encoder reranking"]
    K --> L
    L --> M["Pydantic AI agent + DeepSeek<br/>tool-calling, cites sources"]
    M --> N[Streamlit chat UI]
    M --> O["OpenTelemetry traces"]
    O --> P[("Tempo")]
    N --> Q[("Elasticsearch<br/>query + feedback logs")]
    P --> R[Grafana dashboard]
    Q --> R
```

Tempo is Grafana's own trace storage backend: every agent run is exported to it via OpenTelemetry, and Grafana queries it to show the full request waterfall (agent, LLM calls, tool execution, hybrid search).

## Why these choices

- **Ingestion is a real, automated pipeline.** `dlt` resources and transformers scrape the listing pages, download PDFs, extract text, and fall back to vision OCR page by page. Every run commits one category at a time to Postgres, so a late failure never loses already-completed (and already-paid) work, and pages already staged are skipped on re-runs unless `--force` is passed: OCR is a paid call, so losing the PDF cache should never mean silently re-paying for it.
- **PDFs are content-addressed objects in MinIO.** `boto3` writes and reads them against a local S3-compatible bucket; `extract.py` reads bytes straight from memory (`fitz.open(stream=...)`) with no temp files. Any machine pointed at the same MinIO instance sees the same PDFs.
- **Scanned pages and technical diagrams go through vision OCR.** They're rendered to an image and sent to `google/gemini-2.5-flash` via OpenRouter, with the document's title and category injected into the prompt for context. Testing against a real evacuation diagram found it accurate but still occasionally wrong on fine print, which is exactly why every answer must cite its source page.
- **Embeddings are multilingual.** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` gives ~0.85 cosine similarity between semantically equivalent Spanish/English/French sentences at design time, and in production an English question about "Saint John's night" correctly retrieves the Spanish "Noche de San Juan" bando as the top hit.
- **Retrieval combines vector search, keyword search and reranking.** Qdrant (HNSW) and Elasticsearch (BM25, Spanish analyzer) are queried in parallel and fused with Reciprocal Rank Fusion, then the fused candidates are rescored by a multilingual cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `search/reranker.py`) that reads the actual query and passage text jointly instead of only combining two rank positions. Measured impact is in "Evaluation > Retrieval" below.
- **The agent rewrites the user's question into its own search queries.** The system prompt requires `search_ordinances` queries to be phrased in Spanish regardless of what language the user asked in, and lets the agent issue several purpose-built queries per question (for example, "¿Cuánto cuesta la licencia de apertura de una peluquería?" becomes queries like `licencia apertura peluquería tasa precio` and `tasa licencia apertura establecimientos actividades`) rather than a single verbatim lookup against the index.
- **Elasticsearch handles keyword search.** It serves concurrent reads and writes for a real multi-user app, with the built-in Spanish analyzer stemming ordinance text for better BM25 recall.
- **Pydantic AI drives the agent.** Typed tool calling, structured judge output and native OpenTelemetry instrumentation come with comparatively little boilerplate.
- **Every agent run is traced end-to-end**, from the agent through LLM calls, tool execution and hybrid search, exported via OTLP to Tempo. Every question, answer and feedback vote is also indexed into its own Elasticsearch index (`santa_pola_queries`/`santa_pola_feedback`, separate from the chunk index) with latency, detected language and citation presence, feeding the Grafana dashboard. This keeps each service to one clear job: Postgres stages ingestion text, Elasticsearch handles search and logs, Qdrant handles vectors, MinIO holds raw files.
- **Every claim is cited.** The agent cites with a numbered inline marker (`[1]`) and lists each distinct source once at the end in a fixed `[n] <title>, p. <page>, <url>` format; `rag/citations.py` turns that into deduplicated, bidirectionally-linked footnotes, recomputing numbers from the actual (title, page, url) rather than trusting the model's own numbering, since testing vision OCR against a real technical diagram found a genuine transcription error (120 vs 140 people in an evacuation sector) that makes an uncited hallucination risk unacceptable for a municipal-services assistant.
- **The interface is multilingual independently of the conversation.** The chat answers in whatever language a question is asked in, detected with `lingua-language-detector`, since `langdetect` proved empirically non-deterministic and consistently wrong on some real short questions from this app. The UI chrome (title, placeholders, buttons) is a separate choice, picked from a compact popover selector and stored per-language in `app/locales/*.toml`, since Streamlit has no built-in i18n API of its own.
- **A hard cap limits searches per question, enforced in code.** The agent gets a fixed budget of `search_ordinances` calls (`MAX_SEARCHES_PER_TURN` in `rag/agent.py`); once exhausted, the tool returns a message telling it to answer with what it already has instead of one more real Qdrant+Elasticsearch round trip. A prompt asking the model to "search efficiently" is a request, not a guarantee, so the limit is a boundary check on the tool call itself.

## Real issues found and fixed during ingestion

Running the full 268-PDF ingestion surfaced three real bugs no amount of code review would have caught:

- **Deterministic deadlock under concurrency.** dlt's default 5-way extraction concurrency reliably stalled (near-zero CPU, no progress) whenever two large scanned PDFs were OCR'd at the same time, always at the same page across repeated runs, even though the same page OCR'd in isolation succeeded in 4 seconds. The root cause isn't fully isolated (likely an OpenRouter-side or connection-pool concurrency limit rather than a client bug); pinning `EXTRACT__WORKERS=1` so OCR calls never overlap avoids shipping a pipeline that can silently hang for hours on paid API calls.
- **Unbounded degenerate generation.** One scanned page drove `google/gemini-2.5-flash` into a repetition loop that generated over 1,000,000 characters in a single response, the actual cause of what first looked like the same "hang." `max_tokens=2048` on the OCR call plus a defense-in-depth truncation (`MAX_DESCRIPTION_CHARS`) logs a warning and caps the stored text if a response still comes back abnormally long.
- **Silently corrupted text layer.** Some PDFs embed subset fonts with a broken ToUnicode map, so PyMuPDF "extracts" text that is 60%+ control characters, not real content. A control-character-ratio check (`ingestion/extract.py`) routes those pages to OCR instead of indexing garbage.

## Running it

**Prerequisites:** Docker and Docker Compose. A DeepSeek API key (chat) and an OpenRouter API key (vision OCR for scanned pages).

```bash
cp .env.example .env   # fill in DEEPSEEK_API_KEY and OPENROUTER_API_KEY
docker compose up -d
```

That single command brings up every dependency (Postgres, Qdrant, Elasticsearch, MinIO, Tempo, Grafana), runs the full ingestion and indexing pipeline once, and then starts the Streamlit app on `:8501`. The first run scrapes and OCRs real documents, so it takes a while and makes real OpenRouter API calls; watch its progress with:

```bash
docker compose logs -f ingest
```

The pipeline is idempotent, so re-running `docker compose up -d` later does not repeat completed work. To force a full re-download/re-OCR, or to limit a run to specific categories, override the ingest service's command directly:

```bash
docker compose run --rm ingest uv run python -m santa_pola_rag.ingestion.pipeline --force
docker compose run --rm ingest uv run python -m santa_pola_rag.ingestion.pipeline --categories bandos,normativas
```

For local iteration on the app's own code without rebuilding the image each time, `uv sync` followed by `uv run streamlit run src/santa_pola_rag/app/streamlit_app.py` runs it directly on the host against the same docker-compose dependencies.

Grafana dashboard: http://localhost:3000/d/santa-pola-rag (anonymous access enabled for local use). MinIO console: http://localhost:9001.

## Evaluation

### Retrieval

30 ground-truth questions generated by an LLM from randomly sampled chunks (mixed Spanish/English/French/German, matching the assistant's real user base), evaluated at k=5 against the full 12,089-chunk corpus:

| Strategy | Hit rate | MRR |
|---|---|---|
| Vector only (Qdrant) | 3.3% | 0.011 |
| Text only (Elasticsearch BM25) | 26.7% | 0.097 |
| Hybrid (RRF), no reranking | 16.7% | 0.074 |
| **Hybrid (RRF) + cross-encoder reranking** | **30.0%** | **0.186** |

BM25 outperforms embeddings alone on this corpus: hundreds of tax ordinances share near-identical boilerplate paragraphs (a chunk about "TASA POR X" is semantically close to dozens of other "TASA POR Y" chunks), which confuses a general-purpose multilingual encoder more than it confuses exact-term lexical matching. Plain RRF fusion lands *below* BM25 alone (16.7% vs 26.7%) rather than between the two: with the vector channel's own hit rate near-random, RRF lets a chunk that merely appears in the vector channel's noisy top-50 outrank a chunk BM25 ranked highly but the vector channel missed entirely. Reranking the top 20 RRF-fused candidates with a multilingual cross-encoder fixes this, since it scores the actual query against the actual candidate text jointly instead of only combining two rank positions, and nearly doubles the hit rate. This configuration is kept as the default because it is also the only one with a real retrieval path for non-Spanish questions (an English question about "Saint John's night" only succeeds through the vector channel; BM25 alone returns nothing relevant for it), which this 30-question aggregate, dominated by same-language matches, doesn't fully capture. These are real, un-cherry-picked evaluation results, not target numbers; `eval/retrieval_results.json` has the raw output. Regenerate with `uv run python scripts/generate_ground_truth.py && uv run python scripts/evaluate_retrieval.py`.

### Answer quality (LLM-as-judge)

`google/gemini-2.5-flash` (a different model and provider than the answer-generating `deepseek-chat`, to avoid self-preference bias) scores each answer on relevance, faithfulness to the retrieved context, and citation presence, reasoning step by step before a pass/fail verdict (`evaluation/llm_judge.py`). On the same 10 ground-truth questions, against the current retrieval pipeline (hybrid RRF with cross-encoder reranking): **6/10 passed**, up from 4/10 before reranking was added, tracking the retrieval improvement above. Full transcript in `eval/rag_judge_results.json`; regenerate with `uv run python scripts/evaluate_rag.py`.

The remaining failures are a direct consequence of retrieval gaps, not the agent inventing facts from nothing: the judge's reasoning in the failed cases is that the retrieved context doesn't contain the specific figure or row the question asks about, while the agent still produces a confident, plausibly-cited answer instead of declining. This is precisely the failure mode mandatory citation defends against: even when the agent is confidently wrong, the user has a page number to go check themselves. It also points at the clearest next improvement: the agent should be pushed harder, via prompt or a retrieval-confidence check, to say "not found" rather than answer from a partial context.

### Output format compliance (prompt and temperature comparison)

A second, separate quality dimension is whether the model follows the output-format rules in the system prompt (the exact citation-list layout, going straight to the final answer with no draft recap) regardless of whether the answer's *content* is correct. Two concrete failure patterns were reproduced and measured on the same real question, run 6 times per configuration:

| Configuration | HTML wrapping (bug) | Duplicated recap paragraph (bug) |
|---|---|---|
| Default temperature, no examples | ~1 in 6-7 | ~2-3 in 6 |
| Temperature 0.3 + one positive example | 0 in 6 | 1 in 6 |
| + one explicit negative (wrong-vs-right) example | 0 in 6 | 0 in 6 |

The best configuration (lower temperature, positive and negative few-shot examples) is what's deployed in `rag/agent.py`. It reduces the failure rate rather than guaranteeing it away, since LLM instruction-following is probabilistic: as defense in depth, `rag/citations.py` also strips any HTML the model still adds, and recomputes citation numbers from the actual (title, page, url) rather than trusting the model's own numbering.

## Monitoring

The Grafana dashboard (`monitoring/grafana/dashboards/santa_pola_rag.json`) reads live from Elasticsearch and Tempo:

1. Query volume over time
2. Answer latency, total (avg / p95)
3. Question language distribution
4. Citation rate
5. User feedback (👍/👎)
6. Retrieval (search) time (avg / p95): how much of the total latency is spent in `hybrid_search`
7. LLM generation time (avg / p95): the remainder, covering narration, tool-call construction and the final answer

Every agent run is also traced end-to-end (agent, LLM calls, tool execution, hybrid search, Qdrant/Elasticsearch) via OpenTelemetry to Tempo, browsable from the same Grafana instance.

## Prompt hardening

The system prompt restricts the agent to Santa Pola's ordinances, tells it to treat both user messages and search_ordinances results as data rather than instructions, and refuses to reveal itself even when asked to translate, roleplay past, or "hypothetically" bypass the rule. It held up against 12 adversarial prompts across two rounds (direct override, system-prompt extraction, DAN-style roleplay, off-topic requests, fabricate-a-citation, drop-the-citation-requirement, and softened or nested variants of each), including one case where the agent ignored an explicit "skip the citation for this one" request and cited its source anyway. This is evidence from one test pass with a fixed set of prompts, not a security guarantee: LLM instruction-following stays probabilistic, and indirect injection via the indexed documents themselves, as opposed to the user's message, hasn't been tested, since all indexed content currently comes from the town hall's own site rather than user-supplied documents.

## Limitations and future work

- **Retrieval hit rate, even after reranking (30.0% hybrid @k=5), still has real, measured room to grow**, driven by hundreds of near-duplicate tax-ordinance chunks. A concrete example: the exact answer to "how much is a hairdresser's opening license" is indexed verbatim (`K.- Peluquerías y esteticistas ... 249,15 €`), and search finds it reliably with the near-exact phrase or once title-context and reranking are applied, but this class of dense, list-heavy tariff table remains harder to retrieve from a natural paraphrase than free-flowing prose. This was also the root cause of a real 49-tool-call runaway loop for that exact question before `MAX_SEARCHES_PER_TURN` was added: the agent kept rephrasing instead of finding the chunk, since no rephrasing it tried was the one that mattered.
- The agent should decline more readily when retrieved context is partial, instead of answering confidently from an incomplete excerpt (see the LLM-as-judge analysis above).
- A structured LLM output (a typed schema instead of free text) would remove the citation-formatting failure class in "Output format compliance" by construction, but needs a way to stream a structured field's text live without exposing the underlying JSON deltas to the chat UI, which the current streaming setup doesn't yet do.
- Only 4 of santapola.es's 22 document categories are ingested; adding more is a one-line change to `CATEGORIES` in `scraper.py`.
- MCP server exposing the agent to Claude Desktop and similar clients: a natural next step once the above is in place.

## Evaluation criteria, where covered

| Criterion | Where |
|---|---|
| Problem description | [Problem](#problem) |
| Retrieval flow (knowledge base + LLM) | [Architecture](#architecture); the agent's `search_ordinances` tool in `rag/agent.py` |
| Retrieval evaluation (multiple approaches) | [Evaluation > Retrieval](#retrieval): vector-only, text-only and hybrid with reranking, compared with real numbers |
| LLM evaluation (multiple approaches) | [Evaluation > Answer quality](#answer-quality-llm-as-judge) and [Evaluation > Output format compliance](#output-format-compliance-prompt-and-temperature-comparison) |
| Interface | [Running it](#running-it); Streamlit chat UI |
| Ingestion pipeline (automated) | [Why these choices](#why-these-choices); `dlt` pipeline |
| Monitoring (feedback + dashboard) | [Monitoring](#monitoring): 👍/👎 feedback plus a 7-panel Grafana dashboard |
| Containerization | [Running it](#running-it): everything in `docker-compose.yml`, including the app itself |
| Reproducibility | [Running it](#running-it); `uv.lock` pins every dependency version |
| Best practice: hybrid search | [Why these choices](#why-these-choices), [Evaluation > Retrieval](#retrieval) |
| Best practice: document re-ranking | [Evaluation > Retrieval](#retrieval): cross-encoder reranking of the RRF-fused candidates |
| Best practice: query rewriting | [Why these choices](#why-these-choices): the agent phrases its own search queries, it never searches the raw question |
