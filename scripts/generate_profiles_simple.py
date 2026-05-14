from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from profile_generation.generation import experiment_variants, generate_profile_for_variant
from profile_generation.llm import LLMClient, config_from_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate natural-language user profile variants.")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--run-all", dest="run_all", action="store_true")
    parser.add_argument("--run-hybrid-5way", dest="run_all", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant-id", type=str, default="weighted_x_0.80")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    client = LLMClient(config_from_env("GEN_MODEL", "gorina10.llama3.3:70b"))
    variants = experiment_variants() if args.run_all else [variant for variant in experiment_variants() if variant["variant_id"] == args.variant_id]
    if not variants:
        raise ValueError("Unknown variant_id")

    records = [generate_profile_for_variant(client, args.user_id, variant) for variant in variants]
    payload = {
        "user_id": args.user_id,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} profile records to {args.output}")


if __name__ == "__main__":
    main()