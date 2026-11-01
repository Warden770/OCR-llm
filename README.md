# OCR Attribute Extraction System

An AI-based pipeline that ingests non-standard documents, rasterizes them into page images, and lets Claude do everything else: read the text (OCR, including tables and layout), decide what kind of document it is, decide which attributes the document actually contains, and return each attribute with a confidence score and a page number. There are no predefined field lists or templates. Low-confidence values are routed to human review.

## How extraction works

1. **Ingest and rasterize** PDF, DOCX, DOC and image uploads into page images. Claude reads images (JPEG, PNG, GIF, WebP), and the review screen needs the same page images that regions refer to, so the page images are what Claude reads.
2. **Read with Claude** (`backend/app/claude_extractor.py`). Up to `CLAUDE_PAGES_PER_REQUEST` pages are sent per request as images. Claude returns the document type, a transcription of every page, the attributes it found (name, value, confidence, page) and any tables, as structured JSON (`output_config.format`). Attribute names come from the document itself.
3. **Validate**: empty values, empty labels and out-of-range pages are dropped; duplicates keep the higher confidence.
4. **Route**: confidence of 0.75 or more is pre-approved; everything else goes to human review. A value that does not appear in its own page's transcription is capped at 0.5. This is a consistency check inside a single model response, not an independent verification.
5. **Review** in the dashboard: edit, approve and save attributes to the database, then search and export them. Detected tables appear in the Tables tab, where cells, rows, columns and merges can be edited; a saved table can be edited later and saved again with "Save table changes". Corrections are stored for auditing.

If Claude cannot be reached or returns something unusable (missing or invalid key, rate limit, overload, timeout, refusal, malformed response), the document is marked `FAILED` with a short readable reason, and Retry runs it again. There is no OCR fallback: Tesseract and OpenCV were removed. A request that is cut off by the output limit, or refused, is split in half and retried so that only the offending page fails. An answer that cannot be parsed as JSON (usually a very long one that was cut off) is handled the same way: a multi-page request is split in half, and a page that still fails makes the document `FAILED` with the page number.

### What is different from word-level OCR

- Claude's confidence is its own rating, not a measured accuracy like per-word OCR confidence. Treat it as a review signal.
- **Regions are off by default.** Anthropic documents Claude's pixel coordinates as approximate, and a wrong box is worse than none, so the page viewer shows no highlighted regions unless you set `CLAUDE_REQUEST_BOXES=true`. Attributes stay selectable from the side panel either way. Word-level highlighting and per-word confidence no longer exist.
- Attribute names can vary a little between documents of the same type, because nothing fixes them.
- **Tables.** Claude reads each page visually (borders and lines, cell boundaries, alignment, merged cells, section boundaries) and returns both conventional tables and form-like layouts (bordered sections of labels, headings, long paragraphs and merged cells) as one grid of cells with row and column spans, plus relative column widths and row heights. A form-like table stays one table in the Tables tab and its contents are not repeated as separate text fields; ordinary paragraphs outside a grid are not tables. The layout is a visual reading, so every cell is marked uncertain. Cell boxes exist only when regions are enabled (`CLAUDE_REQUEST_BOXES=true`).

## Stack

- Python 3.10
- Claude API (Messages API, vision and structured outputs) for OCR, document typing and attribute extraction, called with `httpx`
- PyMuPDF (fitz)
- python-docx + LibreOffice for `.doc` conversion, with a direct PNG fallback for `.docx`
- Pillow (PIL)
- FastAPI backend
- Celery worker with Redis broker
- PostgreSQL database
- React + Vite + Tailwind frontend
- Alembic migrations

## Project Layout

- `backend/app/` FastAPI app, ingestion pipeline, Claude extractor, and models
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
| `ANTHROPIC_API_KEY` | Required. Keep it server-side (backend and worker only). If it is missing, documents fail with "ANTHROPIC_API_KEY is not set". |
| `CLAUDE_EXTRACTION_MODEL` | Claude model, default `claude-sonnet-5-5`. Recent models reject forced tool use, which is why the reply is requested as structured JSON. |
| `CLAUDE_PAGES_PER_REQUEST` | Pages sent per request, default `6`. |
| `CLAUDE_IMAGE_MAX_EDGE` | Longest edge in pixels of the JPEG sent for each page, default `2048`. Each page costs up to about 4,800 input tokens for the image, plus the transcription as output. |
| `CLAUDE_TIMEOUT_SECONDS` | Per-request timeout, default `120` (`.env.example` sets `180`). |
| `CLAUDE_REQUEST_DELAY_SECONDS` | Pause between requests for one document, default `0`. Raise it to stay under your rate limits. |
| `CLAUDE_MAX_RETRIES` | How many times a rate limit (429) or server error (5xx) is retried before the request fails, default `5`. |
| `CLAUDE_REQUEST_BOXES` | `true` asks Claude for an approximate region per attribute so the viewer can outline it. Default `false`. |
| `CLAUDE_MAX_OUTPUT_TOKENS`, `CLAUDE_MAX_REQUEST_BYTES` | Optional. Output limit per request (default `16000`) and the image budget per request (default about 25 MB). |
| `SOFFICE_CMD`, `OCR_RASTER_DPI` | LibreOffice command for `.doc` conversion, and the DPI used when rasterizing PDFs (default `300`). |

Rate limits (HTTP 429) and server errors (5xx, including 529 overloaded) are retried up to five times, waiting 2, 4, 8, 16 and 30 seconds (or the `Retry-After` the API sends). If a multi-page request still fails with a server error, it is split in half and each half retried, because shorter answers fail less often when the service is struggling; a single page that still fails makes the document `FAILED` with the status in the message, and Retry runs it again. Error messages carry the HTTP status or reason and Anthropic's `request-id` for support; request payloads, response bodies and the API key are never logged or raised.

**Privacy:** page images are sent to Anthropic for every document. Check the data retention and usage terms that apply to your Anthropic account before using confidential documents. This system has no offline mode.

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

## Checking quality

There is no training and no fixed field list, so quality is checked by reading results. Upload a few representative documents, open them in the review screen, and compare the attributes with the page. Watch for the same concept getting different names across documents of one type, and for values that need correcting; both are visible in the stored corrections. The previous token-level evaluation tooling (CVAT conversion, `predict`, `evaluate`) and the training-data exporter were removed with word-level OCR.
