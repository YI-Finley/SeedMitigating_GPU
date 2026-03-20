#!/usr/bin/env python3
"""
从魔塔(ModelScope)下载模型到本地缓存。
用法:
  python download_modelscope.py --models model_id1 model_id2 --cache_dir /data2/cache/modelscope
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="ModelScope 模型 ID 列表")
    parser.add_argument("--cache_dir", type=str, default="/data2/cache/modelscope", help="缓存目录")
    parser.add_argument("--revision", type=str, default=None, help="可选：模型版本")
    args = parser.parse_args()

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception:
        print("未安装 modelscope，请先安装：pip install modelscope", file=sys.stderr)
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        print(f"下载模型: {model_id}")
        local_dir = snapshot_download(
            model_id,
            cache_dir=str(cache_dir),
            revision=args.revision,
        )
        print(f"完成: {model_id} -> {local_dir}")


if __name__ == "__main__":
    main()
