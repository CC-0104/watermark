"""Boundary sweep for watermark robustness.

This script embeds one payload, applies progressively stronger image attacks,
and writes JSON/HTML reports showing where extraction starts to fail.
It is designed to answer questions such as "how much crop is too much?".
"""

from __future__ import annotations

import argparse
import html
import json
import math
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
    "trustmark": "ORDER-2026-06-29-0001-CUSTOMER-ALPHA-TRACE-BOUNDARY-SWEEP",
    "invismark": "A1",
    "invismark_grid": "AB12",
}


def make_source(path: Path, width: int, height: int, seed: int = 20260629) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[..., 0] = 54 + 160 * x + 18 * np.sin(9 * y)
    img[..., 1] = 72 + 125 * y + 22 * np.cos(7 * x)
    img[..., 2] = 115 + 82 * (1 - x * y) + 26 * np.sin(12 * (x + y))

    # Add a few hard edges and soft areas so the source resembles practical assets
    # more than a pure gradient.
    for idx, color in enumerate(((230, 210, 80), (60, 160, 220), (210, 70, 120))):
        x0 = int(width * (0.12 + idx * 0.21))
        y0 = int(height * (0.16 + idx * 0.13))
        x1 = min(width - 1, x0 + int(width * 0.18))
        y1 = min(height - 1, y0 + int(height * 0.14))
        img[y0:y1, x0:x1] = 0.72 * img[y0:y1, x0:x1] + 0.28 * np.array(color, dtype=np.float32)

    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def parse_size(size: str) -> tuple[int, int]:
    width, height = size.lower().split("x", 1)
    return int(width), int(height)


def relative_uri(path: str | Path, base_dir: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), base_dir.resolve())).as_posix()


def write_image(path: Path, img: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def rotate_keep_canvas(src: np.ndarray, degrees: float) -> np.ndarray:
    if degrees == 0:
        return src.copy()
    height, width = src.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(src, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def apply_attack(src: np.ndarray, family: str, level: float | int, path: Path) -> Path:
    height, width = src.shape[:2]

    if family == "jpeg_quality":
        cv2.imwrite(str(path), src, [int(cv2.IMWRITE_JPEG_QUALITY), int(level)])
        return path

    if family == "center_crop_resize_pct":
        pct = float(level) / 100.0
        y0, y1 = int(height * pct), int(height * (1.0 - pct))
        x0, x1 = int(width * pct), int(width * (1.0 - pct))
        cropped = src[y0:y1, x0:x1]
        restored = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
        return write_image(path, restored)

    if family == "center_crop_keep_pct":
        pct = float(level) / 100.0
        y0, y1 = int(height * pct), int(height * (1.0 - pct))
        x0, x1 = int(width * pct), int(width * (1.0 - pct))
        return write_image(path, src[y0:y1, x0:x1])

    if family == "resize_down_roundtrip_scale":
        scale = float(level)
        small = cv2.resize(
            src,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        return write_image(path, restored)

    if family == "gaussian_blur_kernel":
        kernel = int(level)
        if kernel <= 1:
            return write_image(path, src.copy())
        return write_image(path, cv2.GaussianBlur(src, (kernel, kernel), 0))

    if family == "noise_sigma":
        sigma = float(level)
        if sigma <= 0:
            return write_image(path, src.copy())
        rng = np.random.default_rng(9000 + int(sigma * 10))
        noisy = np.clip(src.astype(np.float32) + rng.normal(0, sigma, src.shape), 0, 255).astype(np.uint8)
        return write_image(path, noisy)

    if family == "rotation_degrees":
        return write_image(path, rotate_keep_canvas(src, float(level)))

    if family == "center_occlusion_area_pct":
        out = src.copy()
        pct = float(level) / 100.0
        if pct > 0:
            side = math.sqrt(pct)
            occ_w = int(width * side)
            occ_h = int(height * side)
            x0 = max(0, (width - occ_w) // 2)
            y0 = max(0, (height - occ_h) // 2)
            out[y0:y0 + occ_h, x0:x0 + occ_w] = (32, 32, 32)
        return write_image(path, out)

    raise ValueError(f"Unknown attack family: {family}")


def attack_plan() -> list[dict]:
    return [
        {
            "family": "jpeg_quality",
            "label": "JPEG quality",
            "unit": "quality",
            "direction": "lower is stronger",
            "levels": [100, 98, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20],
        },
        {
            "family": "center_crop_resize_pct",
            "label": "Center crop then resize",
            "unit": "% each edge",
            "direction": "higher is stronger",
            "levels": [0, 2, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45],
        },
        {
            "family": "center_crop_keep_pct",
            "label": "Center crop only",
            "unit": "% each edge",
            "direction": "higher is stronger",
            "levels": [0, 2, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45],
        },
        {
            "family": "resize_down_roundtrip_scale",
            "label": "Downscale then restore",
            "unit": "scale",
            "direction": "lower is stronger",
            "levels": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.33, 0.25, 0.2],
        },
        {
            "family": "gaussian_blur_kernel",
            "label": "Gaussian blur",
            "unit": "kernel",
            "direction": "higher is stronger",
            "levels": [1, 3, 5, 7, 9, 11, 15, 21],
        },
        {
            "family": "noise_sigma",
            "label": "Gaussian noise",
            "unit": "sigma",
            "direction": "higher is stronger",
            "levels": [0, 1, 2, 3, 5, 8, 10, 15, 20, 30],
        },
        {
            "family": "rotation_degrees",
            "label": "Rotation, same canvas",
            "unit": "degrees",
            "direction": "larger absolute angle is stronger",
            "levels": [0, 1, 2, 3, 5, 8, 10, 15, 30, 45, 90, 180],
        },
        {
            "family": "center_occlusion_area_pct",
            "label": "Center occlusion",
            "unit": "% image area",
            "direction": "higher is stronger",
            "levels": [0, 2, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50],
        },
    ]


def summarize_family(cases: list[dict]) -> dict:
    passed = [case for case in cases if case["ok"]]
    failed = [case for case in cases if not case["ok"]]
    silent = [case for case in cases if case.get("status") == "wrong"]
    stable_until = None
    for case in cases:
        if not case["ok"]:
            break
        stable_until = case["level"]
    return {
        "pass_count": len(passed),
        "total_count": len(cases),
        "stable_until": stable_until,
        "last_pass": passed[-1]["level"] if passed else None,
        "first_fail": failed[0]["level"] if failed else None,
        "silent_failure_count": len(silent),
        "last_pass_image": passed[-1]["image"] if passed else "",
        "first_fail_image": failed[0]["image"] if failed else "",
    }


def run_family(
    wm: InvisibleWatermark,
    method: str,
    text: str,
    embed_result: dict,
    watermarked_path: Path,
    method_dir: Path,
    plan: dict,
) -> dict:
    src = cv2.imread(str(watermarked_path), cv2.IMREAD_COLOR)
    if src is None:
        raise ValueError(f"Cannot read watermarked image: {watermarked_path}")

    family_dir = method_dir / plan["family"]
    family_dir.mkdir(parents=True, exist_ok=True)
    cases = []

    for level in plan["levels"]:
        level_label = str(level).replace(".", "p")
        ext = ".jpg" if plan["family"] == "jpeg_quality" else ".png"
        attack_path = family_dir / f"{plan['family']}_{level_label}{ext}"
        apply_attack(src, plan["family"], level, attack_path)

        start = time.perf_counter()
        try:
            extracted = wm.extract(
                str(attack_path),
                embed_result.get("wm_length", 0),
                robust=method in ("invismark", "invismark_grid", "trustmark"),
            )
            ok = extracted.startswith(text)
            case = {
                "level": level,
                "ok": ok,
                "status": "pass" if ok else "wrong",
                "time_s": round(time.perf_counter() - start, 3),
                "image": str(attack_path),
                "extracted": extracted[:160],
            }
        except Exception as exc:
            case = {
                "level": level,
                "ok": False,
                "status": "error",
                "time_s": round(time.perf_counter() - start, 3),
                "image": str(attack_path),
                "error": str(exc)[:220],
            }
        cases.append(case)

    return {
        "family": plan["family"],
        "label": plan["label"],
        "unit": plan["unit"],
        "direction": plan["direction"],
        "summary": summarize_family(cases),
        "cases": cases,
    }


def run_method(method: str, source_path: Path, workdir: Path, text: str | None) -> dict:
    payload = text or DEFAULT_TEXT.get(method, "A1")
    method_dir = workdir / method
    method_dir.mkdir(parents=True, exist_ok=True)

    wm = InvisibleWatermark(method=method)
    start = time.perf_counter()
    embed_result = wm.embed(str(source_path), str(method_dir / "watermarked.png"), payload)
    embed_s = round(time.perf_counter() - start, 3)
    watermarked_path = Path(embed_result["output"])
    psnr = round(float(InvisibleWatermark.compute_psnr(str(source_path), str(watermarked_path))), 2)

    families = [
        run_family(wm, method, payload, embed_result, watermarked_path, method_dir, plan)
        for plan in attack_plan()
    ]
    return {
        "method": method,
        "payload": payload,
        "embedded_text": embed_result.get("embedded_text", payload),
        "lookup_id": embed_result.get("lookup_id", ""),
        "wm_length": embed_result.get("wm_length"),
        "embed_s": embed_s,
        "psnr": psnr,
        "watermarked": str(watermarked_path),
        "families": families,
    }


def write_html(report: dict, output_path: Path) -> None:
    base_dir = output_path.parent
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Watermark boundary report</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1f2328}",
        "header{padding:28px 36px;background:#18212c;color:#fff}",
        "main{padding:24px 36px}.method{background:#fff;border:1px solid #d8dee4;border-radius:8px;margin-bottom:24px;padding:18px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:14px 0}",
        ".metric{border:1px solid #d8dee4;border-radius:8px;padding:10px;background:#f6f8fa}.metric b{display:block;font-size:22px}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:#fff;margin:12px 0 24px}",
        "th,td{border:1px solid #d8dee4;padding:8px;text-align:left;vertical-align:top}th{background:#f6f8fa}",
        ".ok{color:#116329;font-weight:700}.fail{color:#cf222e;font-weight:700}.wrong{color:#9a6700;font-weight:700}",
        "code{white-space:pre-wrap;word-break:break-word}.thumb{width:150px;max-height:96px;object-fit:contain;background:#111;border-radius:4px}",
        ".bar{height:10px;background:#d8dee4;border-radius:999px;overflow:hidden}.bar span{display:block;height:10px;background:#2da44e}",
        "</style></head><body>",
        "<header><h1>Watermark boundary report</h1>",
        f"<p>Workdir: <code>{html.escape(report['workdir'])}</code></p>",
        f"<p>Source: <code>{html.escape(report['source'])}</code></p></header><main>",
    ]

    for method in report["methods"]:
        parts.append(
            f"<section class='method'><h2>{html.escape(method['method'])}</h2>"
            f"<p>PSNR: <b>{method['psnr']}</b> dB, embed: <b>{method['embed_s']}</b>s, "
            f"embedded text: <code>{html.escape(str(method.get('embedded_text', '')))}</code>, "
            f"lookup id: <code>{html.escape(str(method.get('lookup_id', '')))}</code></p>"
        )
        parts.append("<div class='grid'>")
        for family in method["families"]:
            summary = family["summary"]
            pct = 100.0 * summary["pass_count"] / max(1, summary["total_count"])
            parts.append(
                "<div class='metric'>"
                f"<b>{summary['pass_count']}/{summary['total_count']}</b>"
                f"{html.escape(family['label'])}<br>"
                f"stable until: <code>{html.escape(str(summary['stable_until']))}</code><br>"
                f"first fail: <code>{html.escape(str(summary['first_fail']))}</code><br>"
                f"last any pass: <code>{html.escape(str(summary['last_pass']))}</code><br>"
                f"silent fails: <code>{summary['silent_failure_count']}</code>"
                f"<div class='bar'><span style='width:{pct:.1f}%'></span></div>"
                "</div>"
            )
        parts.append("</div>")

        for family in method["families"]:
            parts.append(
                f"<h3>{html.escape(family['label'])}</h3>"
                f"<p>{html.escape(family['direction'])}; unit: {html.escape(family['unit'])}</p>"
                "<table><thead><tr><th>Level</th><th>Status</th><th>Time</th><th>Sample</th><th>Extract/Error</th></tr></thead><tbody>"
            )
            for case in family["cases"]:
                status = case["status"]
                cls = "ok" if status == "pass" else "wrong" if status == "wrong" else "fail"
                message = case.get("extracted", case.get("error", ""))
                parts.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(case['level']))}</code></td>"
                    f"<td class='{cls}'>{html.escape(status)}</td>"
                    f"<td>{case['time_s']}s</td>"
                    f"<td><img class='thumb' src='{html.escape(relative_uri(case['image'], base_dir))}'></td>"
                    f"<td><code>{html.escape(message)}</code></td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")
        parts.append("</section>")

    parts.append("</main></body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["trustmark"])
    parser.add_argument("--size", default="768x512")
    parser.add_argument("--text", default=None)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="wm_boundary_", dir=ROOT))
    workdir.mkdir(parents=True, exist_ok=True)

    width, height = parse_size(args.size)
    source_path = workdir / f"source_{args.size}.png"
    make_source(source_path, width, height)

    methods = []
    for method in args.methods:
        methods.append(run_method(method, source_path, workdir, args.text))

    report = {
        "workdir": str(workdir),
        "source": str(source_path),
        "size": args.size,
        "methods": methods,
    }
    (workdir / "boundary_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, workdir / "boundary_report.html")
    print(json.dumps({
        "workdir": str(workdir),
        "html_report": str(workdir / "boundary_report.html"),
        "summary": [
            {
                "method": method["method"],
                "psnr": method["psnr"],
                "families": {
                    family["family"]: family["summary"]
                    for family in method["families"]
                },
            }
            for method in methods
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
