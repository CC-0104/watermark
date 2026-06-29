"""Expanded matrix benchmark for watermark methods.

The matrix varies image size and common image attacks, then writes JSON and
HTML reports. It is intended for comparing model wrappers such as original
InvisMark, InvisMark Grid, and TrustMark.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from watermark_engine import InvisibleWatermark  # noqa: E402


DEFAULT_TEXT = {
    "invismark": "A1",
    "invismark_grid": "A1",
    "trustmark": "ORDER-2026-06-29-0001-CUSTOMER-ALPHA-TRACE-JPEG-ROBUSTNESS",
}

DEFAULT_SIZES = ("256x256", "512x512", "768x512", "1024x768")


def parse_size(size: str) -> tuple[int, int]:
    width, height = size.lower().split("x", 1)
    return int(width), int(height)


def make_source(path: Path, width: int, height: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[..., 0] = 70 + 145 * x + 18 * np.sin(12 * y)
    img[..., 1] = 55 + 135 * y + 20 * np.cos(9 * x)
    img[..., 2] = 115 + 75 * (1 - x * y) + 25 * np.sin(14 * (x + y))
    img = np.clip(img + rng.normal(0, 7, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def compute_psnr(source_path: Path, watermarked_path: Path) -> float:
    source = cv2.imread(str(source_path)).astype(np.float64)
    watermarked = cv2.imread(str(watermarked_path)).astype(np.float64)
    if source.shape != watermarked.shape:
        watermarked = cv2.resize(watermarked, (source.shape[1], source.shape[0]))
    mse = np.mean((source - watermarked) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255 ** 2 / mse))


def build_attacks(watermarked_path: Path, workdir: Path) -> dict[str, Path]:
    src = cv2.imread(str(watermarked_path), cv2.IMREAD_COLOR)
    if src is None:
        raise ValueError(f"Cannot read watermarked image: {watermarked_path}")

    height, width = src.shape[:2]
    attacks = {"direct": watermarked_path}

    for quality in (98, 95, 90, 85, 80, 75, 70, 60):
        path = workdir / f"jpeg{quality}.jpg"
        cv2.imwrite(str(path), src, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        attacks[f"jpeg{quality}"] = path

    for scale in (0.5, 0.75, 1.5):
        scaled = cv2.resize(
            src,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        restored = cv2.resize(scaled, (width, height), interpolation=cv2.INTER_CUBIC)
        path = workdir / f"resize{scale}.png"
        cv2.imwrite(str(path), restored)
        attacks[f"resize{scale}x_rt"] = path

    for pct in (5, 10, 20):
        crop = src[
            int(height * pct / 100): int(height * (1 - pct / 100)),
            int(width * pct / 100): int(width * (1 - pct / 100)),
        ]
        restored = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
        path = workdir / f"crop{pct}.png"
        cv2.imwrite(str(path), restored)
        attacks[f"crop{pct}_resize"] = path

    path = workdir / "rot90.png"
    cv2.imwrite(str(path), cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE))
    attacks["rotate90"] = path

    path = workdir / "blur3.png"
    cv2.imwrite(str(path), cv2.GaussianBlur(src, (3, 3), 0))
    attacks["blur3"] = path

    rng = np.random.default_rng(999)
    for sigma in (2, 5):
        noisy = np.clip(src.astype(np.int16) + rng.normal(0, sigma, src.shape), 0, 255).astype(np.uint8)
        path = workdir / f"noise{sigma}.png"
        cv2.imwrite(str(path), noisy)
        attacks[f"noise{sigma}"] = path

    return attacks


def run_one(method: str, size_label: str, source_path: Path, workdir: Path) -> dict:
    text = DEFAULT_TEXT.get(method, "A1")
    method_dir = workdir / f"{method}_{size_label}"
    method_dir.mkdir(parents=True, exist_ok=True)
    row = {"method": method, "size": size_label, "text": text}

    wm = InvisibleWatermark(method=method)
    start = time.perf_counter()
    embed = wm.embed(str(source_path), str(method_dir / "watermarked.png"), text)
    row["embed_s"] = round(time.perf_counter() - start, 3)
    row["psnr"] = round(compute_psnr(source_path, Path(embed["output"])), 2)
    row["embedded_text"] = embed.get("embedded_text", text)

    cases = []
    for attack_name, attack_path in build_attacks(Path(embed["output"]), method_dir).items():
        start = time.perf_counter()
        try:
            extracted = wm.extract(
                str(attack_path),
                embed.get("wm_length", 0),
                robust=method in ("invismark", "invismark_grid", "trustmark"),
            )
            ok = extracted.startswith(text)
            cases.append({
                "case": attack_name,
                "ok": ok,
                "time_s": round(time.perf_counter() - start, 3),
                "extracted": extracted[:120],
            })
        except Exception as exc:
            cases.append({
                "case": attack_name,
                "ok": False,
                "time_s": round(time.perf_counter() - start, 3),
                "error": str(exc)[:160],
            })

    row["pass_count"] = sum(1 for case in cases if case["ok"])
    row["total_count"] = len(cases)
    row["failed"] = [case["case"] for case in cases if not case["ok"]]
    row["cases"] = cases
    return row


def write_html(report: dict, path: Path) -> None:
    rows = report["summary"]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Watermark matrix benchmark</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f6f7f9;color:#1f2328}",
        "table{border-collapse:collapse;width:100%;background:white;border:1px solid #d8dee4}",
        "th,td{border:1px solid #d8dee4;padding:8px;text-align:left;vertical-align:top;font-size:13px}",
        "th{background:#f6f8fa}.ok{color:#116329;font-weight:700}.fail{color:#cf222e;font-weight:700}",
        "code{white-space:pre-wrap;word-break:break-word}",
        "</style></head><body>",
        "<h1>Watermark matrix benchmark</h1>",
        f"<p>Workdir: <code>{html.escape(report['workdir'])}</code></p>",
        "<table><thead><tr><th>Method</th><th>Size</th><th>PSNR</th><th>Embed s</th><th>Pass</th><th>Failed cases</th></tr></thead><tbody>",
    ]
    for row in rows:
        if "error" in row:
            failed = row["error"]
            pass_text = "ERROR"
            pass_class = "fail"
        else:
            failed = ", ".join(row.get("failed", [])) or "none"
            pass_text = f"{row.get('pass_count')}/{row.get('total_count')}"
            pass_class = "ok" if row.get("pass_count") == row.get("total_count") else "fail"
        parts.append(
            "<tr>"
            f"<td>{html.escape(row['method'])}</td>"
            f"<td>{html.escape(row['size'])}</td>"
            f"<td>{html.escape(str(row.get('psnr', '')))}</td>"
            f"<td>{html.escape(str(row.get('embed_s', '')))}</td>"
            f"<td class='{pass_class}'>{html.escape(pass_text)}</td>"
            f"<td><code>{html.escape(failed)}</code></td>"
            "</tr>"
        )
    parts.append("</tbody></table></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["invismark", "invismark_grid", "trustmark"])
    parser.add_argument("--sizes", nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="wm_matrix_", dir=ROOT))
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for size_label in args.sizes:
        width, height = parse_size(size_label)
        source_path = workdir / f"source_{size_label}.png"
        make_source(source_path, width, height, width + height)
        for method in args.methods:
            try:
                results.append(run_one(method, size_label, source_path, workdir))
            except Exception as exc:
                results.append({"method": method, "size": size_label, "error": str(exc)})

    summary = [
        {key: result.get(key) for key in ("method", "size", "psnr", "embedded_text", "embed_s", "pass_count", "total_count", "failed", "error") if key in result}
        for result in results
    ]
    report = {"workdir": str(workdir), "summary": summary, "results": results}
    (workdir / "matrix_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, workdir / "matrix_report.html")
    print(json.dumps({"workdir": str(workdir), "summary": summary, "html_report": str(workdir / "matrix_report.html")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
