# Document Workflow

Claude reads the page images, decides the document type and which attributes the document contains, and returns
each attribute with a confidence score and page number. Nothing is trained and there are no field lists.

## Folder Layout

```text
D:\OCR\
  datasets/
    raw-pages/                 Generated PNG pages (one folder per document id)
      <document_id>/
        page-001.png
        page-002.png
  uploads/                     Original uploaded files; runtime storage
```

## Running a document

### 1. Start the system

Set `ANTHROPIC_API_KEY` in your `.env` first (see the README configuration table), then start the backend:

```powershell
cd D:\OCR
python -m uvicorn app.main:app --reload --app-dir backend --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```powershell
cd D:\OCR\frontend
npm run dev -- --host localhost --port 5173
```

Open:

- Dashboard: http://localhost:5173/
- API: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs/

### 2. Upload a document

Upload through the dashboard. The system creates a document ID, for example `42`, stores the original as
`uploads/42/source.pdf`, writes the page images to `datasets/raw-pages/42/`, and sends those images to Claude.

If Claude fails, the processing screen shows the reason (for example a bad key or a rate limit) and Retry runs
the document again.

### 3. Review

Open the document, check the highlighted regions against the page, edit labels and values, select the
attributes to keep and save them. Values below 75% confidence, and values Claude could not find in its own
transcription of the page, are marked for review.

## Fresh Start Cleanup

To begin again, stop the backend and remove only generated/user data:

```powershell
Remove-Item -Recurse -Force datasets\raw-pages\*
Remove-Item -Recurse -Force uploads\*
Remove-Item -Force ocr_system.db, backend\ocr_system.db -ErrorAction SilentlyContinue
```

Keep the source code and the README files.

## Online Multi-User Deployment

Before exposing the system online, configure these server-only values in the deployment environment:

```text
AUTH_REQUIRED=true
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_JWT_SECRET=<server-only JWT signing secret>
SUPABASE_SERVICE_ROLE_KEY=<server-only service-role key>
SUPABASE_STORAGE_BUCKET=ocr-documents
```

Create a private Supabase Storage bucket named `ocr-documents`. Never put the service-role key in the frontend. Set these public frontend values separately:

```text
VITE_SUPABASE_URL=https://<project-ref>.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=<publishable-or-anon-key>
```

The backend and Celery worker use the Supabase database, Redis, and private Storage. The browser must sign in with Supabase Auth. Every document is assigned to the authenticated user's ID, and document, page, attribute, correction, search, export, and image requests are owner-scoped. RLS is also enabled in Supabase as defense in depth.

Test deployment isolation with two Supabase Auth accounts: user A must not be able to list, open, search, edit, export, delete, or retrieve page images belonging to user B.
