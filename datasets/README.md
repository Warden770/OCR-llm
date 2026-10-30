# Dataset Workspace

Use these folders only:

```text
datasets/
  raw-pages/                 PNG pages used by the application, CVAT, and prediction
  cvat/                      CVAT export files, usually annotations.xml
  ground-truth/
    source/                  JSON files authored by hand, if used
    ground_truth.jsonl       converted CVAT annotations
  predictions/               LLM output, predictions.jsonl
```

Do not put PDFs, CVAT exports, or predictions in `uploads/`. Uploaded PDFs remain there; processed page images belong in `raw-pages/`.

Recommended flow:

1. Run a PDF through the OCR application. It automatically stores the resulting page PNGs in `raw-pages/<document_id>/`.
2. Upload those exact PNGs to CVAT and annotate the fields.
3. Save the CVAT XML export in `cvat/annotations.xml`.
4. Convert it to `ground-truth/ground_truth.jsonl`.
5. Generate predictions into `predictions/predictions.jsonl`.
6. Evaluate predictions against `ground-truth/ground_truth.jsonl`.
7. Adjust the prompt or schemas if needed and repeat steps 5-6.

Ground truth is independent human annotation. Never replace it with model predictions.
