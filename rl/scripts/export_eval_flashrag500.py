#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_BASE = Path("/root/code/SeedMitigating/output/eval_flashrag500_7models")
DEFAULT_ROOT = DEFAULT_BASE / "response_level"
DEFAULT_OUT_DIR = DEFAULT_BASE / "report"
DEFAULT_OUT_XLSX = DEFAULT_OUT_DIR / "eval_flashrag500_7models_metrics.xlsx"
DEFAULT_OUT_PPT = DEFAULT_OUT_DIR / "eval_flashrag500_7models_report.pptx"
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified export entrypoint for XLSX and PPTX."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Input response_level directory. Default: %(default)s",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for derived files. Default: %(default)s",
    )
    parser.add_argument(
        "--xlsx",
        dest="do_xlsx",
        action="store_true",
        help="Export xlsx only.",
    )
    parser.add_argument(
        "--ppt",
        dest="do_ppt",
        action="store_true",
        help="Export pptx only.",
    )
    parser.add_argument(
        "--summary",
        dest="do_summary",
        action="store_true",
        help="Export summary txt only.",
    )
    return parser.parse_args()


def run_cmd(cmd):
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = any([args.do_xlsx, args.do_ppt, args.do_summary])
    do_xlsx = args.do_xlsx or not selected
    do_ppt = args.do_ppt or not selected
    do_summary = args.do_summary

    script_dir = Path(__file__).resolve().parent
    py = sys.executable

    if do_xlsx:
        run_cmd([
            py,
            str(script_dir / "export_eval_flashrag500_tables.py"),
            "--root", str(root),
            "--out", str(out_dir / DEFAULT_OUT_XLSX.name),
        ])

    if do_ppt or do_summary:
        cmd = [
            py,
            str(script_dir / "export_eval_flashrag500_report.py"),
            "--root", str(root),
            "--out-ppt", str(out_dir / DEFAULT_OUT_PPT.name),
            "--out-summary", str(out_dir / "analysis_summary.txt"),
        ]
        if not do_ppt:
            cmd.append("--skip-ppt")
        if not do_summary:
            cmd.append("--skip-summary")
        run_cmd(cmd)


if __name__ == "__main__":
    main()
