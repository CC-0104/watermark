"""End-to-end demo for TrustMark + watermark_id + SQLite registry."""

from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from watermark_engine import InvisibleWatermark  # noqa: E402
from watermark_registry import DEFAULT_DB_PATH, WatermarkRegistry  # noqa: E402


DEFAULT_PAYLOAD = (
    "ORDER-2026-06-29-0001;CUSTOMER=ALPHA;CHANNEL=JPEG70;"
    "RIGHTS=INTERNAL;TRACE=INDUSTRIAL-DEMO"
)


def make_source(path: Path, width: int = 768, height: int = 512) -> None:
    rng = np.random.default_rng(2026)
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[..., 0] = 65 + 140 * x + 16 * np.sin(10 * y)
    img[..., 1] = 80 + 120 * y + 18 * np.cos(8 * x)
    img[..., 2] = 120 + 70 * (1 - x * y) + 20 * np.sin(12 * (x + y))
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def compute_psnr(source_path: Path, watermarked_path: Path) -> float:
    source = cv2.imread(str(source_path)).astype(np.float64)
    watermarked = cv2.imread(str(watermarked_path)).astype(np.float64)
    mse = np.mean((source - watermarked) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10 * np.log10(255 ** 2 / mse))


def db_stats(db_path: Path) -> dict:
    if not db_path.exists():
        return {"exists": False}
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM watermark_records").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM watermark_records WHERE status = 'active'").fetchone()[0]
    return {"exists": True, "records": count, "active_records": active, "size_bytes": db_path.stat().st_size}


def write_html(report: dict, output_path: Path) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>TrustMark registry demo</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f6f7f9;color:#1f2328}",
        ".panel{background:white;border:1px solid #d8dee4;border-radius:8px;padding:18px;margin-bottom:18px}",
        "table{border-collapse:collapse;width:100%}th,td{border-top:1px solid #d8dee4;padding:8px;text-align:left;vertical-align:top}",
        "th{background:#f6f8fa}.ok{color:#116329;font-weight:700}.fail{color:#cf222e;font-weight:700}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}",
        "img{max-width:100%;background:#111;border-radius:6px}code{white-space:pre-wrap;word-break:break-word}",
        "</style></head><body>",
        "<h1>TrustMark + watermark_id + SQLite registry</h1>",
    ]
    parts.append("<section class='panel'><h2>End-to-end result</h2><table>")
    for key in ("payload_text", "watermark_id", "psnr", "jpeg70_ok", "resolved_payload", "registry_path"):
        value = report.get(key)
        cls = "ok" if key == "jpeg70_ok" and value else "fail" if key == "jpeg70_ok" else ""
        parts.append(f"<tr><th>{html.escape(key)}</th><td class='{cls}'><code>{html.escape(str(value))}</code></td></tr>")
    parts.append("</table></section>")

    parts.append("<section class='panel'><h2>Registry record</h2><table>")
    record = report.get("registry_record", {})
    for key, value in record.items():
        parts.append(f"<tr><th>{html.escape(key)}</th><td><code>{html.escape(str(value))}</code></td></tr>")
    parts.append("</table></section>")

    parts.append("<section class='panel'><h2>Image artifacts</h2><div class='grid'>")
    for label, rel in (("source", "source.png"), ("watermarked", "watermarked.png"), ("jpeg70", "jpeg70.jpg")):
        parts.append(f"<figure><img src='{rel}' alt='{label}'><figcaption>{label}</figcaption></figure>")
    parts.append("</div></section>")
    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", default=DEFAULT_PAYLOAD)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="wm_registry_demo_", dir=ROOT))
    workdir.mkdir(parents=True, exist_ok=True)

    source = workdir / "source.png"
    make_source(source)

    wm = InvisibleWatermark(method="trustmark")
    result = wm.embed(str(source), str(workdir / "watermarked.png"), args.payload)
    watermarked = Path(result["output"])

    img = cv2.imread(str(watermarked))
    jpeg70 = workdir / "jpeg70.jpg"
    cv2.imwrite(str(jpeg70), img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    resolved = wm.extract(str(jpeg70), robust=True)
    registry = WatermarkRegistry(args.db)
    record = registry.get(result["lookup_id"]) if result.get("lookup_id") else None

    report = {
        "payload_text": args.payload,
        "watermark_id": result.get("lookup_id") or result.get("embedded_text"),
        "embedded_text": result.get("embedded_text"),
        "resolved_payload": resolved,
        "jpeg70_ok": resolved == args.payload,
        "psnr": round(compute_psnr(source, watermarked), 2),
        "registry_path": str(args.db),
        "registry_stats": db_stats(args.db),
        "registry_record": record or {},
        "workdir": str(workdir),
    }
    (workdir / "registry_demo.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, workdir / "registry_demo.html")
    print(json.dumps({**report, "html_report": str(workdir / "registry_demo.html")}, ensure_ascii=False, indent=2))
    return 0 if report["jpeg70_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
