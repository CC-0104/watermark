"""System robustness smoke test for the watermark engines.

This script intentionally avoids pytest so it can be run in a fresh checkout:

    python tests/watermark_system_test.py

It reports attack outcomes as JSON. By default only a failed direct round trip
causes a non-zero exit code; robustness cases are measurements, not hard gates.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from watermark_engine import InvisibleWatermark  # noqa: E402


DEFAULT_TEXT = {
    "adaptive_dwt": "Watermark-2026",
    "trustmark": "ORDER-2026-06-29-0001-CUSTOMER-ALPHA-TRACE-JPEG-ROBUSTNESS",
    "invismark": "A1",
    "invismark_pro": "A1",
    "invismark_logpolar": "A1",
    "invismark_grid": "AB12",
    "dwt_dct": "Watermark-2026",
    "dwt_dct_svd": "Watermark-2026",
}


def make_source(path: Path, width: int = 768, height: int = 512) -> None:
    rng = np.random.default_rng(123)
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[..., 0] = 80 + 120 * x + 20 * np.sin(10 * y)
    img[..., 1] = 60 + 130 * y + 15 * np.cos(8 * x)
    img[..., 2] = 100 + 80 * (1 - x * y) + 20 * np.sin(18 * (x + y))
    img = np.clip(img + rng.normal(0, 7, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def build_attacks(watermarked_path: Path, workdir: Path) -> dict[str, Path]:
    rng = np.random.default_rng(456)
    src = cv2.imread(str(watermarked_path), cv2.IMREAD_COLOR)
    if src is None:
        raise ValueError(f"Cannot read watermarked image: {watermarked_path}")

    height, width = src.shape[:2]
    attacks = {"direct": watermarked_path}

    for quality in (95, 85, 70):
        path = workdir / f"jpeg{quality}.jpg"
        cv2.imwrite(str(path), src, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        attacks[f"jpeg{quality}"] = path

    small = cv2.resize(src, (int(width * 0.75), int(height * 0.75)), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    path = workdir / "resize075_roundtrip.png"
    cv2.imwrite(str(path), resized)
    attacks["resize075_roundtrip"] = path

    path = workdir / "blur3.png"
    cv2.imwrite(str(path), cv2.GaussianBlur(src, (3, 3), 0))
    attacks["blur3"] = path

    noisy = np.clip(src.astype(np.int16) + rng.normal(0, 2, src.shape), 0, 255).astype(np.uint8)
    path = workdir / "noise2.png"
    cv2.imwrite(str(path), noisy)
    attacks["noise2"] = path

    path = workdir / "rotate90.png"
    cv2.imwrite(str(path), cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE))
    attacks["rotate90"] = path

    crop = src[int(height * 0.05): int(height * 0.95), int(width * 0.05): int(width * 0.95)]
    crop_rt = cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)
    path = workdir / "crop5_resize.png"
    cv2.imwrite(str(path), crop_rt)
    attacks["crop5_resize"] = path

    return attacks


def compute_ssim(source: np.ndarray, watermarked: np.ndarray) -> float:
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float64)
    watermarked_gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(source_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(watermarked_gray, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(source_gray * source_gray, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(watermarked_gray * watermarked_gray, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(source_gray * watermarked_gray, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(np.mean(ssim_map))


def save_diff_image(source_path: Path, watermarked_path: Path, output_path: Path) -> dict:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    watermarked = cv2.imread(str(watermarked_path), cv2.IMREAD_COLOR)
    if source is None or watermarked is None:
        raise ValueError("Cannot read source or watermarked image for diff")
    if source.shape != watermarked.shape:
        watermarked = cv2.resize(watermarked, (source.shape[1], source.shape[0]))

    diff = cv2.absdiff(source, watermarked)
    diff_float = diff.astype(np.float64)
    changed_pixels = np.any(diff > 0, axis=2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    amplified = np.clip(gray.astype(np.float32) * 16.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(amplified, cv2.COLORMAP_MAGMA)
    cv2.imwrite(str(output_path), heatmap)

    return {
        "ssim": round(compute_ssim(source, watermarked), 6),
        "mae": round(float(np.mean(diff_float)), 4),
        "rmse": round(float(np.sqrt(np.mean(diff_float ** 2))), 4),
        "max_abs_diff": int(np.max(diff)),
        "changed_pixel_pct": round(float(np.mean(changed_pixels) * 100.0), 4),
    }


def run_method(method: str, source_path: Path, workdir: Path) -> dict:
    text = DEFAULT_TEXT.get(method, "Watermark-2026")
    method_dir = workdir / method
    method_dir.mkdir(parents=True, exist_ok=True)

    wm = InvisibleWatermark(method=method, key=123456)
    output_path = method_dir / "watermarked.png"
    embed_result = wm.embed(str(source_path), str(output_path), text)
    watermarked_path = Path(embed_result["output"])
    diff_path = method_dir / "diff_amplified.png"
    visual_metrics = save_diff_image(source_path, watermarked_path, diff_path)
    psnr = InvisibleWatermark.compute_psnr(str(source_path), str(watermarked_path))

    rows = []
    for attack_name, attack_path in build_attacks(watermarked_path, method_dir).items():
        try:
            extracted = wm.extract(
                str(attack_path),
                embed_result.get("wm_length", 0),
                robust=method.startswith("invismark"),
            )
            ok = extracted.startswith(text)
            rows.append({"case": attack_name, "ok": ok, "image": str(attack_path), "extracted": extracted})
        except Exception as exc:  # Robustness failures should be visible in the report.
            rows.append({"case": attack_name, "ok": False, "image": str(attack_path), "error": str(exc)})

    return {
        "method": method,
        "text": text,
        "source": str(source_path),
        "watermarked": str(watermarked_path),
        "diff": str(diff_path),
        "psnr": round(float(psnr), 2),
        "visual_metrics": visual_metrics,
        "wm_length": embed_result.get("wm_length"),
        "embedded_text": embed_result.get("embedded_text", text),
        "lookup_id": embed_result.get("lookup_id", ""),
        "rows": rows,
    }


def relative_uri(path: str | Path, base_dir: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), base_dir.resolve())).as_posix()


def write_html_report(report: dict, output_path: Path) -> None:
    base_dir = output_path.parent
    methods = report["results"]
    total_cases = sum(len(result.get("rows", [])) for result in methods)
    passed_cases = sum(1 for result in methods for row in result.get("rows", []) if row.get("ok"))

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Watermark robustness report</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1f2328}",
        "header{padding:28px 36px;background:#17202a;color:white}",
        "main{padding:24px 36px}",
        ".summary{display:flex;gap:16px;flex-wrap:wrap;margin-top:14px}",
        ".metric{background:#263445;border-radius:8px;padding:12px 16px;min-width:150px}",
        ".metric b{display:block;font-size:24px;margin-bottom:4px}",
        ".method{background:white;border:1px solid #d8dee4;border-radius:8px;margin:0 0 24px;padding:20px}",
        ".method h2{margin:0 0 6px;font-size:22px}",
        ".meta{color:#57606a;margin-bottom:16px}",
        ".compare{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:18px}",
        ".figure{border:1px solid #d8dee4;border-radius:8px;overflow:hidden;background:#fff}",
        ".figure img{width:100%;display:block;object-fit:contain;background:#111;max-height:280px}",
        ".figure figcaption{padding:8px 10px;font-size:13px;color:#57606a}",
        "table{border-collapse:collapse;width:100%;font-size:14px}",
        "th,td{border-top:1px solid #d8dee4;text-align:left;padding:9px;vertical-align:top}",
        "th{background:#f6f8fa;color:#57606a}",
        ".ok{color:#116329;font-weight:700}.fail{color:#cf222e;font-weight:700}",
        ".thumb{width:128px;max-height:86px;object-fit:contain;background:#111;border-radius:4px}",
        "code{white-space:pre-wrap;word-break:break-word}",
        "</style></head><body>",
        "<header>",
        "<h1>Watermark robustness report</h1>",
        "<div class='summary'>",
        f"<div class='metric'><b>{html.escape(str(len(methods)))}</b>methods</div>",
        f"<div class='metric'><b>{passed_cases}/{total_cases}</b>attack cases passed</div>",
        f"<div class='metric'><b>{html.escape(str(len(report['hard_failures'])))}</b>hard failures</div>",
        "</div></header><main>",
    ]

    for result in methods:
        if "error" in result:
            parts.append("<section class='method'>")
            parts.append(f"<h2>{html.escape(result['method'])}</h2>")
            parts.append(f"<p class='fail'>{html.escape(result['error'])}</p>")
            parts.append("</section>")
            continue

        parts.append("<section class='method'>")
        parts.append(f"<h2>{html.escape(result['method'])}</h2>")
        parts.append(
            "<div class='meta'>"
            f"payload: <code>{html.escape(result['text'])}</code> | "
            f"embedded: <code>{html.escape(result.get('embedded_text', result['text']))}</code> | "
            f"PSNR: {html.escape(str(result['psnr']))} dB | "
            f"bits: {html.escape(str(result['wm_length']))}"
            "</div>"
        )
        metrics = result.get("visual_metrics", {})
        parts.append(
            "<table><thead><tr><th colspan='6'>Visual impact</th></tr>"
            "<tr><th>PSNR</th><th>SSIM</th><th>MAE</th><th>RMSE</th><th>Max diff</th><th>Changed px</th></tr></thead><tbody>"
            "<tr>"
            f"<td>{html.escape(str(result['psnr']))} dB</td>"
            f"<td>{html.escape(str(metrics.get('ssim', '')))}</td>"
            f"<td>{html.escape(str(metrics.get('mae', '')))}</td>"
            f"<td>{html.escape(str(metrics.get('rmse', '')))}</td>"
            f"<td>{html.escape(str(metrics.get('max_abs_diff', '')))}</td>"
            f"<td>{html.escape(str(metrics.get('changed_pixel_pct', '')))}%</td>"
            "</tr></tbody></table>"
        )
        parts.append("<div class='compare'>")
        for label, key in (("Source", "source"), ("Watermarked", "watermarked"), ("Amplified difference", "diff")):
            uri = relative_uri(result[key], base_dir)
            parts.append(
                "<figure class='figure'>"
                f"<img src='{html.escape(uri)}' alt='{html.escape(label)}'>"
                f"<figcaption>{html.escape(label)}</figcaption>"
                "</figure>"
            )
        parts.append("</div>")
        parts.append("<table><thead><tr><th>Attack</th><th>Preview</th><th>Status</th><th>Extracted / error</th></tr></thead><tbody>")
        for row in result["rows"]:
            status = "<span class='ok'>PASS</span>" if row["ok"] else "<span class='fail'>FAIL</span>"
            message = row.get("extracted", row.get("error", ""))
            uri = relative_uri(row["image"], base_dir)
            parts.append(
                "<tr>"
                f"<td>{html.escape(row['case'])}</td>"
                f"<td><img class='thumb' src='{html.escape(uri)}' alt='{html.escape(row['case'])}'></td>"
                f"<td>{status}</td>"
                f"<td><code>{html.escape(message)}</code></td>"
                "</tr>"
            )
        parts.append("</tbody></table></section>")

    parts.append("</main></body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["adaptive_dwt", "invismark_grid", "trustmark"])
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--keep", action="store_true", help="Keep generated images")
    parser.add_argument("--report", action="store_true", help="Write report.html and report.json, and keep assets")
    parser.add_argument("--html-report", type=Path, help="Custom HTML report path")
    parser.add_argument("--strict", action="store_true", help="Fail if any attack fails")
    args = parser.parse_args()

    created_tmp = args.workdir is None
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="wm_system_test_", dir=ROOT))
    workdir.mkdir(parents=True, exist_ok=True)

    source_path = workdir / "source.png"
    make_source(source_path)

    results = []
    hard_failures = []
    for method in args.methods:
        try:
            result = run_method(method, source_path, workdir)
            results.append(result)
            direct = next((row for row in result["rows"] if row["case"] == "direct"), None)
            if not direct or not direct["ok"]:
                hard_failures.append(f"{method}: direct round trip failed")
            if args.strict:
                for row in result["rows"]:
                    if not row["ok"]:
                        hard_failures.append(f"{method}: {row['case']} failed")
        except Exception as exc:
            results.append({"method": method, "error": str(exc)})
            hard_failures.append(f"{method}: {exc}")

    report = {
        "workdir": str(workdir),
        "results": results,
        "hard_failures": hard_failures,
    }

    if args.report or args.html_report:
        html_path = args.html_report or (workdir / "report.html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        json_path = html_path.with_name("report.json")
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_html_report(report, html_path)
        report["html_report"] = str(html_path)
        report["json_report"] = str(json_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    keep_outputs = args.keep or args.report or args.html_report
    if created_tmp and not keep_outputs:
        shutil.rmtree(workdir, ignore_errors=True)

    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
