# OCR Attribute Extraction System

An AI-based pipeline that ingests non-standard documents, rasterizes them at 300 DPI, classifies the document type, reads the text with Tesseract, and extracts the fields defined for that document type with a multimodal LLM (Claude or Gemini). Every extracted value is tied back to the OCR tokens it came from, and low-confidence fields are routed to human review.

## How extraction works

1. **Ingest and rasterize** PDF, DOCX, DOC and image uploads into page images.
2. **OCR** each page with Tesseract to get tokens with bounding boxes.
3. **Classify** the document type (keyword scoring) to select the attribute schema in `backend/app/document_schemas.py`.
4. **Extract** with the LLM (`backend/app/llm_extractor.py`). The model receives the page image and the numbered OCR tokens, and returns the schema fields through a forced tool call. Each field cites the token indices it came from, so bounding boxes are exact and the review UI can highlight the source.
5. **Route**: confidence of 0.75 or more is pre-approved; everything else goes to human review. Values that cite no OCR tokens are capped at 0.5, and values that differ from the cited OCR text (the model corrected an OCR error) are scaled down, so both are reviewed by a person.
6. **Review** in the dashboard. Corrections are stored for auditing and for building an evaluation set.

There is no model training. Behaviour is changed by editing the prompt in `llm_extractor.py` or the field lists in `document_schemas.py`. If `ANTHROPIC_API_KEY` is not set, or an LLM call fails, the page falls back to the OCR-only heuristics in `routing_engine.py`, so uploads never block on the API.

## Stack

- Python 3.10
- Anthropic API (Claude) or Gemini API, both multimodal, for extraction
- Tesseract OCR v5
- OpenCV
- PyMuPDF (fitz)
- python-docx + LibreOffice for `.doc` conversion, with a direct PNG fallback for `.docx`
- Pillow (PIL)
- FastAPI backend
- Celery worker with Redis broker
- PostgreSQL database
- React + Vite + Tailwind frontend
- Alembic migrations

## Project Layout

- `backend/app/` FastAPI app, ingestion pipeline, OCR processor, inference, models, and routing
- `backend/alembic/` database migration environment
- `frontend/` HITL dashboard UI
- `datasets/raw-pages/` rasterized page storage for the application, CVAT, and evaluation
- `uploads/` raw upload staging directory

## Local Development

1. Copy [.env.example](.env.example) to `.env` and adjust values if needed, or use the provided local `.env` for SQLite.
2. Install Python dependencies with `pip install -r requirements.txt`.
3. Install frontend dependencies with `cd frontend && npm install`.
4. If you want the full Docker stack, start infrastructure with `docker compose up redis`. Set `DATABASE_URL` to PostgreSQL when using a PostgreSQL service; otherwise the backend uses the explicit local SQLite default.
5. Run migrations with `alembic -c backend/alembic.ini upgrade head` if using PostgreSQL. SQLite will create tables automatically.
6. Start the backend with `python scripts/check_port_free.py --port 8000 && python -m uvicorn app.main:app --reload --app-dir backend`.
7. Start the Celery worker with `celery -A app.tasks.celery_app worker --loglevel=info --app-dir backend`.
8. Start the frontend with `cd frontend && npm run dev`.

If port 8000 is already occupied, the preflight check exits with a clear message: `Port 8000 already in use by PID X — kill it first or choose another port.`

## Configuration

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Enables LLM extraction. Keep it server-side (backend and worker only). |
| `GEMINI_API_KEY` | Enables the Gemini backend. A free key from Google AI Studio works; a Google AI Pro/Plus subscription does not include API access. |
| `GEMINI_EXTRACTION_MODEL`, `GEMINI_TIMEOUT_SECONDS` | Gemini model (default `gemini-2.5-flash`) and request timeout (60). |
| `EXTRACTION_BACKEND` | `llm` (Claude), `gemini` or `ocr`. Blank means Claude if `ANTHROPIC_API_KEY` is set, else Gemini if `GEMINI_API_KEY` is set, else `ocr`. |
| `LLM_EXTRACTION_MODEL` | Model ID, default `claude-sonnet-5-5`. |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | Request timeout (120) and retry count (3). |

Privacy note: with an LLM backend, page images and OCR text are sent to Anthropic or Google. Gemini free-tier requests may be used by Google to improve its products, so do not use the free tier for confidential documents. Use `EXTRACTION_BACKEND=ocr` for documents that must not leave your environment.

## Run The System

To run the full system locally:

1. Start the database and Redis if you are using Docker-based infrastructure: `docker compose up db redis`.
2. Start the backend API from the repository root: `uvicorn app.main:app --reload --app-dir backend`.
3. Start the Celery worker in a second terminal: `celery -A app.tasks.celery_app worker --loglevel=info --app-dir backend`.
4. Start the frontend in a third terminal: `cd frontend && npm run dev`.
5. Open the dashboard at `http://localhost:5173` and the API docs at `http://localhost:8000/docs`.

Set `DATABASE_URL` explicitly to select PostgreSQL or SQLite. A configured PostgreSQL connection failure stops startup instead of switching to a different database, preventing split state and apparent data loss.

Celery runs in asynchronous mode by default. Set `CELERY_TASK_ALWAYS_EAGER=true` in your environment if you want synchronous execution for local debugging or tests.

DOCX files are currently rasterized with a direct text-to-PNG fallback in the ingestion pipeline. LibreOffice/PDF conversion is used for `.doc` inputs.

## Measuring accuracy

Because nothing is trained, ground truth is used only to measure how well the prompt and schemas work. Build it from independent human annotation, never from the model's own output:

```bash
# From the backend directory
python -m app.dataset_tools cvat-to-jsonl --xml ../datasets/cvat/annotations.xml --image-root ../datasets/raw-pages --output ../datasets/ground-truth/ground_truth.jsonl
python -m app.dataset_tools predict --raw ../datasets/raw-pages --output ../datasets/predictions/predictions.jsonl
python -m app.dataset_tools evaluate --predictions ../datasets/predictions/predictions.jsonl --ground-truth ../datasets/ground-truth/ground_truth.jsonl
```

`predict` runs OCR and the LLM extractor over page images and writes token-level labels. `evaluate` reports token accuracy, bounding-box IoU match rate, per-label precision/recall/F1, and optional field exact-match rate. Re-run it after any prompt or schema change to see whether accuracy moved. If you tune the prompt on some documents, keep a separate set for the final check.

CVAT rectangle exports are converted into the ground-truth format. See [cvat/README.md](cvat/README.md) for the self-hosted CVAT workflow. CVAT is an annotation tool here, not part of the extraction path. Approved and corrected attributes can also be exported with `python -m app.exporter output/export.jsonl` for use as additional evaluation data.
