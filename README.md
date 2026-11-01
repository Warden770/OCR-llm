# OCR Attribute Extraction System

An AI-based pipeline that ingests non-standard documents, rasterizes them into page images, and lets Gemini do everything else: read the text (OCR, including tables and layout), decide what kind of document it is, decide which attributes the document actually contains, and return each attribute with a confidence score and a page number. There are no predefined field lists or templates. Low-confidence values are routed to human review.

## How extraction works

1. **Ingest and rasterize** PDF, DOCX, DOC and image uploads into page images. Gemini cannot read `.docx`, `.doc`, `.tiff` or `.bmp` directly, and the review screen needs the same page images that regions refer to, so the page images are what Gemini reads.
2. **Read with Gemini** (`backend/app/gemini_extractor.py`). Up to `GEMINI_PAGES_PER_REQUEST` pages are sent per request as images. Gemini returns the document type, a transcription of every page, the attributes it found (name, value, confidence, page, approximate region) and any tables, as structured JSON. Attribute names come from the document itself.
3. **Validate**: empty values, empty labels and out-of-range pages are dropped; duplicates keep the higher confidence; regions are converted to the app's `[x0, y0, x1, y1]` boxes on a 0..1000 scale.
4. **Route**: confidence of 0.75 or more is pre-approved; everything else goes to human review. A value that does not appear in its own page's transcription is capped at 0.5. This is a consistency check inside a single model response, not an independent verification.
5. **Review** in the dashboard: edit, approve and save attributes to the database, then search and export them. Detected tables appear in the Tables tab, where cells, rows, columns and merges can be edited; a saved table can be edited later and saved again with "Save table changes". Corrections are stored for auditing.

Gemini sometimes refuses to reproduce text it has memorized (finish reason `RECITATION`), which happens with standard wording such as university vision and mission statements. The extractor then splits the request to find the page that triggers it and retries that page with less verbatim text: first without the full-page transcription (tables are kept), and only if that is stopped too, attributes only. Attributes from such a page still come back, but they cannot be checked against a transcription, so they are capped at 0.5 confidence and go to review.

If Gemini cannot be reached or returns something unusable (missing or invalid key, rate limit, timeout, blocked or malformed response), the document is marked `FAILED` with a short readable reason, and Retry runs it again. There is no OCR fallback: Tesseract and OpenCV were removed.

### What is different from word-level OCR

- Gemini's confidence is its own rating, not a measured accuracy like per-word OCR confidence. Treat it as a review signal.
- Regions are approximate. Word-level highlighting and per-word confidence no longer exist; the page viewer outlines each attribute's region instead.
- Attribute names can vary a little between documents of the same type, because nothing fixes them.
- **Tables.** Gemini reads each page visually (borders and lines, cell boundaries, alignment, merged cells, section boundaries) and returns both conventional tables and form-like layouts (bordered sections of labels, headings, long paragraphs and merged cells) as one grid of cells with row and column spans, plus relative column widths and row heights. A form-like table stays one table in the Tables tab and its contents are not repeated as separate text fields; ordinary paragraphs outside a grid are not tables. The layout is a visual reading, so every cell is marked uncertain. Cell boxes are derived from the table box when there is one.

## Stack

- Python 3.10
- Gemini API (multimodal) for OCR, document typing and attribute extraction, called with `httpx`
- PyMuPDF (fitz)
- python-docx + LibreOffice for `.doc` conversion, with a direct PNG fallback for `.docx`
- Pillow (PIL)
- FastAPI backend
- Celery worker with Redis broker
- PostgreSQL database
- React + Vite + Tailwind frontend
- Alembic migrations

## Project Layout

- `backend/app/` FastAPI app, ingestion pipeline, Gemini extractor, and models
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
| `GEMINI_API_KEY` | Required. A free key from Google AI Studio works; a Google AI Pro/Plus subscription does not include API access. Keep it server-side (backend and worker only). If it is missing, documents fail with "GEMINI_API_KEY is not set". |
| `GEMINI_EXTRACTION_MODEL` | Gemini model, default `gemini-3.5-flash`. |
| `GEMINI_PAGES_PER_REQUEST` | Pages sent per request, default `6`. A request that is cut off by the output limit is split in half and retried. |
| `GEMINI_IMAGE_MAX_EDGE` | Longest edge in pixels of the JPEG sent for each page, default `2048`. |
| `GEMINI_TIMEOUT_SECONDS` | Per-request timeout, default `120` (`.env.example` sets `180`). |
| `GEMINI_REQUEST_DELAY_SECONDS` | Pause between requests for one document, default `0`. Raise it to stay under free-tier rate limits. |
| `GEMINI_MAX_OUTPUT_TOKENS`, `GEMINI_MAX_REQUEST_BYTES` | Optional. Output limit per request (default `32768`) and the inline image budget per request (default about 18 MB). |
| `SOFFICE_CMD`, `OCR_RASTER_DPI` | LibreOffice command for `.doc` conversion, and the DPI used when rasterizing PDFs (default `300`). |

Rate limits (HTTP 429) and server errors (5xx) are retried up to three times with exponential backoff, honouring `Retry-After`. Error messages carry the HTTP status or reason only; request payloads, response bodies and the API key are never logged or raised.

**Privacy:** page images are sent to Google for every document. Gemini free-tier requests may be used by Google to improve its products, so do not use the free tier for confidential documents. This system has no offline mode.

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
