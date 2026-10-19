# Bi-encoder retrieval vs cross-encoder re-ranking on the GDPR

A RAG pipeline over one long legal document, comparing two ways of choosing the 3 passages given to the LLM:

- **Pipeline 1:** bi-encoder top 10 from FAISS, first 3 by cosine similarity
- **Pipeline 2:** the same top 10 re-scored by a cross-encoder, first 3 by cross-encoder score

Both pipelines use the same chunks, the same prompt and temperature 0, so any difference between their answers comes from retrieval.

All code, outputs and comments are in **`rag_assignment.ipynb`**.

## Document

Regulation (EU) 2016/679 (GDPR), official PDF from EUR-Lex, 88 pages (`gdpr.pdf`).
It has 173 recitals (background, not binding) and 99 articles (the rules). The two parts share vocabulary, so recitals often compete with the article that actually answers a question.

Text is extracted with PyMuPDF. Footnotes are removed by font size (8.5 pt vs 9.6 pt body) and page headers by pattern. The text is split at recital/article boundaries, and anything over 200 words is cut into 200-word windows with 40 words of overlap. Result: 434 chunks (`chunks.csv`).

## Setup

| Part | Choice |
|---|---|
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalised) |
| Vector store | FAISS `IndexFlatIP` (exact cosine) |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Gemini via `google-genai`, temperature 0, thinking off |
| Hardware | MacBook Air (Apple Silicon), CPU only |

Gemini free-tier quotas are per model per day, so the notebook falls back through `gemini-3.8-flash` → `3.7-flash` → `3.5-flash`. The failure-case answers came from `gemini-3.7-flash` (K1 and K3 from `gemini-3.5-flash`), the latency calls from `gemini-3.8-flash`. The two answers for one query always come from the same model, and the model used is stored with each answer in `results/`.

## Results

29 test queries in four groups (keyword trap, exception, negation, recital vs article). Each has a gold phrase that appears only in the passage that answers it.

| | Pipeline 1 | Pipeline 2 |
|---|---|---|
| Hit@1 | 0.241 | 0.310 |
| Hit@3 (gold chunk reaches the LLM) | 0.483 | 0.586 |
| MRR@10 | 0.375 | 0.435 |

Rank shifts where re-ranking changed what the LLM saw:

| Query | Gold chunk | Rank (bi → cross) |
|---|---|---|
| E1 Does a company always have to inform affected people about a data breach? | Art 34(3) exemptions | 8 → 3 |
| N11 Is no special approval needed to transfer data to a country the Commission considers adequate? | Art 45(1) | 7 → 3 |
| N13 Can the data protection officer be fired for doing their job? | Art 38(3) | 5 → 3 |
| K1 Can I make a company delete everything it knows about me? | Art 17(1) | 4 → 3 |
| K3 What is the maximum penalty a company can get for violating the regulation? | Art 83 fine amounts | 2 → 4 (worse) |

In 10 of 29 queries the gold chunk was not in the bi-encoder's top 10, so re-ranking could not recover it.

Latency (median of 290 runs, CPU):

| Step | ms |
|---|---|
| Query embedding | 10.9 |
| FAISS search, 434 chunks | 0.04 |
| Cross-encoder, 10 pairs | 157.9 |
| Pipeline 1 retrieval | 11.0 |
| Pipeline 2 retrieval | 169.0 |
| Gemini call (gemini-3.8-flash) | ~4,250-4,500 |

The cross-encoder on all 434 chunks takes 4.7 s per query (10.9 ms per pair). Projected to 1,000,000 chunks that is about 3 hours per query, against about 44 ms for bi-encoder retrieval plus about 160 ms for re-ranking the top 10.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo 'GEMINI_API_KEY=your-key' > .env
jupyter notebook rag_assignment.ipynb
```

Gemini answers are cached in `results/llm_cache.json`, so re-running the notebook reproduces the same answers without API calls. The latency cell is the exception: it always calls the API. Delete the cache file to get fresh answers.

## Files

- `rag_assignment.ipynb`: the full pipeline and analysis, with outputs
- `gdpr.pdf`: source document
- `chunks.csv`: the 434 chunks with section and page
- `results/`: query sweep, retrieval metrics, per-query answers, latency and scaling tables, LLM cache
