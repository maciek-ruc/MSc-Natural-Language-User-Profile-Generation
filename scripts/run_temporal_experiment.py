from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from profile_generation.llm import LLMClient, config_from_env
from profile_generation.temporal_experiment import run_experiment_for_user, select_users


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the temporal profile evaluation pipeline.")
    parser.add_argument("--users", type=str, default="")
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--holdout-n", type=int, default=2)
    parser.add_argument("--holdout-publications", type=int, default=2)
    parser.add_argument("--min-train-interactions", type=int, default=5)
    parser.add_argument("--max-publications", type=int, default=30)
    parser.add_argument("--max-interaction-examples", type=int, default=30)
    parser.add_argument("--shared-evidence-cap", type=int, default=30)
    parser.add_argument("--min-interacted-articles", type=int, default=1)
    parser.add_argument("--fixed-priority-source-cap", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    users = select_users(
        args.users,
        args.max_users,
        args.holdout_n,
        args.min_train_interactions,
        max(args.holdout_publications, 2),
    )
    gen_client = LLMClient(config_from_env("GEN_MODEL", "gorina10.llama3.3:70b"))
    judge_client = LLMClient(config_from_env("JUDGE_MODEL", "gpt-4.1-mini"))

    results = []
    for user_id in users:
        print(f"Running user {user_id}", flush=True)
        try:
            results.append(
                run_experiment_for_user(
                    user_id=user_id,
                    gen_client=gen_client,
                    judge_client=judge_client,
                    holdout_n=args.holdout_n,
                    holdout_publications=args.holdout_publications,
                    max_publications=args.max_publications,
                    max_interaction_examples=args.max_interaction_examples,
                    shared_evidence_cap=args.shared_evidence_cap,
                    min_interacted_articles=args.min_interacted_articles,
                    fixed_priority_source_cap=args.fixed_priority_source_cap,
                    top_k=args.top_k,
                )
            )
        except Exception as exc:
            results.append({"user_id": user_id, "status": "failed", "error": str(exc)})

    payload = {
        "user_count": len(users),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()