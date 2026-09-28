#!/usr/bin/env python3
"""Split a ShareGPT JSONL into contiguous shards plus a small smoke shard."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--smoke-rows", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise SystemExit("num-shards must be greater than 0")
    if not args.input.is_file():
        raise SystemExit(f"input does not exist: {args.input}")

    with args.input.open(encoding="utf-8") as handle:
        row_count = sum(1 for line in handle if line.strip())
    if row_count == 0:
        raise SystemExit(f"input is empty: {args.input}")

    per_shard = math.ceil(row_count / args.num_shards)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [
        args.out_dir / f"shard-{index:04d}.jsonl" for index in range(args.num_shards)
    ]
    handles = [path.open("w", encoding="utf-8") for path in shard_paths]
    smoke_path = args.out_dir / "shard-smoke.jsonl"
    smoke_written = 0
    try:
        with args.input.open(encoding="utf-8") as src, smoke_path.open(
            "w", encoding="utf-8"
        ) as smoke:
            index = 0
            for line in src:
                if not line.strip():
                    continue
                if not line.endswith("\n"):
                    line = line + "\n"
                shard_index = min(index // per_shard, args.num_shards - 1)
                handles[shard_index].write(line)
                if smoke_written < args.smoke_rows:
                    smoke.write(line)
                    smoke_written += 1
                index += 1
    finally:
        for handle in handles:
            handle.close()

    (args.out_dir / "num_shards").write_text(f"{args.num_shards}\n", encoding="utf-8")
    (args.out_dir / "row_count").write_text(f"{row_count}\n", encoding="utf-8")
    print(
        f"split {row_count} rows into {args.num_shards} shards "
        f"(~{per_shard} each), smoke={smoke_written}"
    )


if __name__ == "__main__":
    main()
