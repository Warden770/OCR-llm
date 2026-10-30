# CVAT Annotation Workflow

CVAT is used only to create independent ground-truth annotations. The model must not generate the labels used as ground truth.

## Start CVAT

Use the official CVAT self-hosted deployment for the current release:

```bash
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

Open `http://localhost:8080`, create a task, upload the raw page images, and create rectangle labels such as `TOTAL`, `DATE`, or `VENDOR`. Export the task as **CVAT for images 1.1 XML**.

Keep the exported XML and source images outside the model prediction folder. Convert the export from this repository root:

```bash
python -m app.dataset_tools cvat-to-jsonl \
  --xml cvat-export/annotations.xml \
  --image-root datasets/raw-pages \
  --output datasets/ground-truth/ground_truth.jsonl
```

Run that command from `backend`, or set `PYTHONPATH=backend` when running it from the repository root. The converter runs the existing Tesseract OCR over each image and assigns a CVAT rectangle label to tokens whose rectangle overlap is at least 10%.

For best results, annotate complete field spans and keep the image files unchanged between CVAT and conversion. Review `datasets/ground-truth/ground_truth.jsonl` before using it for evaluation, especially for words that cross rectangle boundaries.
