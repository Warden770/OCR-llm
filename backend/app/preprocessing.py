"""Document image preprocessing utilities for OCR.

Provides a standalone `DocumentPreprocessor` helper that can prepare 300 DPI
PNG page images for Tesseract v5.

This module is not wired into the live OCR path in `ocr_processor.py`. The
active pipeline currently uses a simpler grayscale → Gaussian blur → adaptive
threshold → deskew sequence, while this helper keeps a more experimental
grayscale → deskew → denoise → CLAHE → adaptive threshold workflow available
for offline use or future integration.

This module is pure (no side effects) except optional image file writes in
`process()` when `output_path` is provided, and is safe to call from a Celery
worker. Each stage is guarded against exceptions and falls back to the
previous image when a stage fails.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, TypedDict, cast

import cv2
from cv2.typing import MatLike
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class _ProcessResult(TypedDict):
    processed_image: MatLike | None
    skew_angle: float
    output_path: str | None
    processing_time_ms: float


class DocumentPreprocessor:
    """Preprocess single-page document images for OCR.

    Parameters are chosen to be conservative for 300 DPI scanned pages:
    - Small Gaussian kernel (3x3) reduces sensor noise but preserves edges.
    - CLAHE `clipLimit=2.0` and `tileGridSize=(8,8)` are common defaults for
      document photography to boost local contrast without amplifying noise.
    - Adaptive threshold uses a relatively large block size to capture local
      background variations while preserving text.
    """

    def __init__(self, use_nl_means: bool = False) -> None:
        self.use_nl_means = use_nl_means

    def to_grayscale(self, image: MatLike | None) -> MatLike:
        """Convert an image to single-channel grayscale.

        Accepts BGR or RGB arrays and returns an 8-bit grayscale image.
        """
        if image is None:
            raise ValueError("Input image is None")
        if len(image.shape) == 2:
            return image.copy()
        if image.shape[2] == 3:
            # Heuristic: assume BGR if uint8 and common OpenCV usage, but both
            # BGR and RGB convert to the same luminance with cvtColor using
            # COLOR_BGR2GRAY. If images were RGB, result is still acceptable.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return gray
        if image.shape[2] == 4:
            # Drop alpha channel
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            return gray
        raise ValueError(f"Unsupported image shape: {image.shape}")

    def deskew(self, image: MatLike) -> tuple[MatLike, float]:
        """Detect and correct skew using Hough line transform on Canny edges.

        Returns (rotated_image, angle_degrees). If no dominant lines are found
        the original image is returned with angle 0.0.
        """
        h, w = image.shape[:2]
        gray = image if len(image.shape) == 2 else self.to_grayscale(image)

        edges = cv2.Canny(gray, 50, 150)
        # HoughLinesP parameters scaled to page size
        min_line_length = max(30, w // 8)
        lines = cast(
            MatLike | None,
            cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=80,
                minLineLength=min_line_length,
                maxLineGap=10,
            ),
        )

        angles: list[float] = []
        if lines is not None and len(lines) > 0:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                ang = np.degrees(np.arctan2(dy, dx))
                # Normalize large angles: prefer angles in [-90,90]
                if ang > 180:
                    ang -= 360
                if ang < -180:
                    ang += 360
                # Correct angles outside +/-45 to avoid accidental 90deg flips
                if ang > 45:
                    ang -= 90
                elif ang < -45:
                    ang += 90
                angles.append(ang)

        # If Hough didn't yield useful angles, try a fallback using minAreaRect
        if not angles:
            try:
                # Binary image for component geometry
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                coords = cast(MatLike | None, cv2.findNonZero(255 - bw))  # text pixels (non-white)
                if coords is not None and len(coords) > 0:
                    rect = cv2.minAreaRect(coords)
                    # rect[2] is angle in degrees in range [-90, 0)
                    rect_angle = rect[2]
                    # Convert to a conventional angle: positive means CCW
                    if rect_angle > 45:
                        rect_angle = rect_angle - 90
                    elif rect_angle < -45:
                        rect_angle = rect_angle + 90
                    median_angle = float(-rect_angle)
                else:
                    return image, 0.0
            except Exception:
                return image, 0.0
        else:
            # Use median for robustness against outliers
            median_angle = float(np.median(angles))

        # Rotate to correct skew (negative of detected angle)
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, -median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated, median_angle

    def denoise(self, image: MatLike | None) -> MatLike:
        """Reduce sensor/scan noise while preserving text edges.

        By default uses a small Gaussian blur (3x3). When `use_nl_means` is
        True, uses `cv2.fastNlMeansDenoising` which is stronger and slower.
        """
        if image is None:
            raise ValueError("Input image is None")
        gray = image if len(image.shape) == 2 else self.to_grayscale(image)
        if self.use_nl_means:
            try:
                den = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
                return den
            except Exception:
                logger.warning("fastNlMeansDenoising failed, falling back to Gaussian blur")
        # Conservative Gaussian blur to reduce noise without destroying edges
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        return denoised

    def apply_clahe(
        self,
        image: MatLike,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> MatLike:
        """Apply CLAHE (adaptive histogram equalization) to the grayscale image.

        This helps with uneven lighting and shadows from phone-camera scans.
        """
        gray = image if len(image.shape) == 2 else self.to_grayscale(image)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        try:
            res = clahe.apply(gray)
            return res
        except Exception:
            logger.warning("CLAHE failed; returning original image")
            return gray

    def adaptive_threshold(self, image: MatLike, invert: Optional[bool] = None) -> MatLike:
        """Convert the grayscale image to a binary image using adaptive thresholding.

        If `invert` is None the method will attempt to auto-detect inversion by
        checking the mean intensity (dark background vs light background).
        """
        gray = image if len(image.shape) == 2 else self.to_grayscale(image)
        # blockSize must be odd and tuned for 300 DPI documents; use ~35
        block_size = 35 if (gray.shape[0] > 200 or gray.shape[1] > 200) else 15
        C = 10
        try:
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, C)
        except Exception:
            logger.warning("adaptiveThreshold failed; falling back to global Otsu")
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Auto-detect inversion: if most pixels are dark, invert to make text dark
        if invert is None:
            white_ratio = np.mean(th > 0)
            # For typical documents, white_ratio should be >0.5 (background white)
            if white_ratio < 0.5:
                invert = True
            else:
                invert = False

        if invert:
            th = cv2.bitwise_not(th)

        return th

    def process(self, image_path: str, output_path: Optional[str] = None) -> _ProcessResult:
        """Run the full preprocessing pipeline on `image_path`.

        Returns a dict with keys:
        - `processed_image`: numpy.ndarray or None if loading failed
        - `skew_angle`: float (degrees)
        - `output_path`: str or None
        - `processing_time_ms`: float

        Each stage is wrapped in try/except; failures log a warning and fall back
        to the previous image state.
        """
        t0 = time.time()
        skew_angle = 0.0
        processed: MatLike | None = None

        if not os.path.exists(image_path):
            logger.warning("Image path does not exist: %s", image_path)
            return {
                "processed_image": None,
                "skew_angle": 0.0,
                "output_path": None,
                "processing_time_ms": 0.0,
            }

        try:
            img = cast(MatLike | None, cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
            if img is None:
                raise IOError("cv2.imread returned None")
        except Exception as e:
            logger.warning("Failed to load image %s: %s", image_path, e)
            return {
                "processed_image": None,
                "skew_angle": 0.0,
                "output_path": None,
                "processing_time_ms": 0.0,
            }

        # Stage: grayscale
        try:
            t_start = time.time()
            gray = self.to_grayscale(img)
            logger.info("to_grayscale: %.1f ms", (time.time() - t_start) * 1000)
        except Exception as e:
            logger.warning("to_grayscale failed: %s", e)
            gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Stage: deskew
        try:
            t_start = time.time()
            deskewed, angle = self.deskew(gray)
            skew_angle = angle
            logger.info("deskew: %.1f ms (angle=%.3f)", (time.time() - t_start) * 1000, angle)
        except Exception as e:
            logger.warning("deskew failed: %s", e)
            deskewed = gray

        # Stage: denoise
        try:
            t_start = time.time()
            den = self.denoise(deskewed)
            logger.info("denoise: %.1f ms", (time.time() - t_start) * 1000)
        except Exception as e:
            logger.warning("denoise failed: %s", e)
            den = deskewed

        # Stage: CLAHE
        try:
            t_start = time.time()
            clahe = self.apply_clahe(den)
            logger.info("clahe: %.1f ms", (time.time() - t_start) * 1000)
        except Exception as e:
            logger.warning("apply_clahe failed: %s", e)
            clahe = den

        # Stage: adaptive threshold
        try:
            t_start = time.time()
            th = self.adaptive_threshold(clahe, invert=None)
            logger.info("adaptive_threshold: %.1f ms", (time.time() - t_start) * 1000)
        except Exception as e:
            logger.warning("adaptive_threshold failed: %s", e)
            th = clahe if len(clahe.shape) == 2 else self.to_grayscale(clahe)

        processed = th

        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                cv2.imwrite(output_path, processed)
            except Exception as e:
                logger.warning("Failed to write processed image to %s: %s", output_path, e)
                output_path = None

        processing_time_ms = (time.time() - t0) * 1000.0
        result: _ProcessResult = {
            "processed_image": processed,
            "skew_angle": skew_angle,
            "output_path": output_path,
            "processing_time_ms": processing_time_ms,
        }
        return result

    def batch_process(
        self,
        image_paths: List[str],
        output_dir: str,
        max_workers: int = 4,
    ) -> List[_ProcessResult]:
        """Process multiple images in parallel and return list of result dicts.

        Output files are written to `output_dir` preserving original filenames.
        """
        os.makedirs(output_dir, exist_ok=True)
        results: List[_ProcessResult] = []

        def _task(p: str) -> _ProcessResult:
            out_name = os.path.basename(p)
            out_path = os.path.join(output_dir, out_name)
            return self.process(p, out_path)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_task, p): p for p in image_paths}
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    logger.warning("batch task failed for %s: %s", futures[f], e)
                    results.append({"processed_image": None, "skew_angle": 0.0, "output_path": None, "processing_time_ms": 0.0})

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a single document image for OCR")
    parser.add_argument("input", help="Input image path (PNG)")
    parser.add_argument("output", nargs='?', help="Output path for processed image")
    args = parser.parse_args()

    pre = DocumentPreprocessor()
    res = pre.process(args.input, args.output)
    if res["processed_image"] is None:
        print("Processing failed or image unreadable")
    else:
        print(f"Processed in {res['processing_time_ms']:.1f} ms; skew_angle={res['skew_angle']:.3f}")
