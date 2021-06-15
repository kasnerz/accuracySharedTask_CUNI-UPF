#!/usr/bin/env python3
"""
Merge JSONs generated in parallel by generate.py
"""
import os
import argparse
import logging
import json
import glob

from utils.json_encoder import CompactJSONEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
        help="Directory with JSONs.")
    parser.add_argument("--outdir", type=str, required=True,
        help="Output directory.")
    parser.add_argument("--prefix", type=str, required=True,
        help="Prefix to merge.")
    args = parser.parse_args()

    files = glob.glob(f"{args.dir}/{args.prefix}-*.json")
    data = []

    for filename in files:
        with open(filename) as f:
            j = json.load(f)
            data += j["data"]

    with open(f"{args.outdir}/{args.prefix}.json", "w") as f:
        s = json.dumps({"data" : data}, ensure_ascii=False, indent=4, cls=CompactJSONEncoder)
        f.write(s)

