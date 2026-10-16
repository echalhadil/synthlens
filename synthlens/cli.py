from __future__ import annotations

import argparse
import base64
import json
import os
import sys

from .core.detector import detect

BARS = 20


def _bar(p: float) -> str:
    filled = int(round(p * BARS))
    return "[" + "#" * filled + "-" * (BARS - filled) + "]"


def _text_report(result) -> str:
    lines = []
    lines.append(f"Verdict:     {result.verdict}")
    lines.append(f"Probability: {result.probability:.2f}  AI  {_bar(result.probability)}")
    lines.append("")
    lines.append("Heuristics:")
    ordered = sorted(result.heuristics.items(),
                     key=lambda kv: kv[1].weighted, reverse=True)
    for name, h in ordered:
        lines.append(f"  {name:<14} score={h.score:.2f} weight={h.weight:.2f} "
                     f"contrib={h.weighted:.3f}")
    return "\n".join(lines)


def _save_viz(result, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name, h in result.heuristics.items():
        for viz_name, b64 in h.viz.items():
            if not b64:
                continue
            path = os.path.join(out_dir, f"{name}_{viz_name}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            written.append(path)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="synthlens",
        description="Detect AI-generated images with no-ML forensic analysis.",
    )
    parser.add_argument("image", help="path to the image file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show per-heuristic breakdown")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="output result as JSON")
    parser.add_argument("--save-viz", metavar="DIR", default=None,
                        help="write analysis visualizations to DIR")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.image):
        print(f"error: file not found: {args.image}", file=sys.stderr)
        return 2

    result = detect(args.image, include_viz=bool(args.save_viz))

    if args.save_viz:
        files = _save_viz(result, args.save_viz)
        for f in files:
            print(f"viz: {f}", file=sys.stderr)

    if args.as_json:
        print(json.dumps(result.to_dict(include_viz=False), indent=2))
    elif args.verbose:
        print(_text_report(result))
    else:
        print(f"{result.verdict}  (p={result.probability:.2f})  {args.image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())