# CVAT

CVAT was used to build word-level ground truth for the Tesseract and token-based evaluation tools. Those
tools (`dataset_tools cvat-to-jsonl`, `predict`, `evaluate`) were removed when Gemini became the only OCR and
document-understanding engine, because Gemini returns attributes with approximate regions rather than word
tokens. There is nothing in this repository that consumes CVAT exports any more.

CVAT itself is a separate application and can still be used to annotate pages if you build a new evaluation
tool around attribute-level ground truth.
