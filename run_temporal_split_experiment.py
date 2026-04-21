"""Run strict temporal-split experiment for hybrid NL profile variants.

Pipeline per user:
1) Build train/holdout split from article interaction timeline.
2) Generate configured hybrid profile variants from train-only evidence.
3) Judge each variant on holdout titles.
4) Store split, candidates, judge scores, and top-k selection in experiment_* tables.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arxivdigest.core.database.connection import get_connection
from scripts.generate_user_nl_profiles import (
    LLMClient,
    LLMConfig,
    _generate_profile_from_hybrid_evidence,
    _hybrid_variant_specs,
    _resolve_effective_weight,
)
from scripts.llm_connector import LLMConnectionConfig
from scripts.llm_connector import LLMJudgeClient
from scripts.llm_connector import load_project_env
from scripts.llm_connector import pick_first_nonempty
from scripts.llm_connector import resolve_api_key


def _normalize_profile_for_dedup(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return normalized


def _parse_weighted_x_values(raw: str) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for piece in (raw or "").split(","):
        text = piece.strip()
        if not text:
            continue
        value = _resolve_effective_weight(float(text))
        rounded = round(value, 4)
        if rounded in seen:
            continue
        seen.add(rounded)
        values.append(value)
    if not values:
        values = [0.2, 0.5, 0.8]
    return values


def _experiment_variants(weighted_x_values: list[float]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for value in weighted_x_values:
        spec = dict(_hybrid_variant_specs(value)["weighted"])
        suffix = f"{value:.2f}"
        spec["variant_id"] = f"weighted_x_{suffix}"
        spec["prompt_variant"] = f"weighted_x_{suffix}"
        variants.append(spec)

    priority_specs = _hybrid_variant_specs(0.5)
    variants.append(dict(priority_specs["prompt_publications_priority"]))
    variants.append(dict(priority_specs["prompt_interactions_priority"]))
    return variants


_VAGUE_FOCUS_TERMS = {"resource", "resources", "benchmark", "benchmarks", "automatic", "comprehensive", "knowledge"}


def _apply_judge_quality_guard(judge: dict[str, Any], profile_text: str) -> dict[str, Any]:
    guarded = dict(judge or {})
    text = (profile_text or "").lower()

    focus_match = re.search(r"with a focus on\s+([^\.\n]+)", text)
    focus_phrase = (focus_match.group(1).strip() if focus_match else "")
    focus_words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", focus_phrase)

    severe_vague_focus = False
    if focus_words and len(focus_words) <= 2:
        if all(w in _VAGUE_FOCUS_TERMS for w in focus_words):
            severe_vague_focus = True

    if severe_vague_focus:
        guarded["specificity"] = min(float(guarded.get("specificity", 5)), 2.0)
        guarded["consistency"] = min(float(guarded.get("consistency", 5)), 2.0)
        guarded["overall"] = min(float(guarded.get("overall", 5)), 2.0)
        guarded["pass_fail"] = "fail"
        prev = str(guarded.get("rationale_short", "")).strip()
        suffix = "Auto-quality-guard: vague focus phrase with low semantic specificity."
        guarded["rationale_short"] = (prev + " | " + suffix).strip(" |")[:500]

    return guarded


def _select_users(users_arg: str, max_users: int, holdout_n: int, min_train: int) -> list[int]:
    if users_arg.strip():
        return [int(x.strip()) for x in users_arg.split(",") if x.strip()]

    with closing(get_connection()) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            """
            SELECT af.user_id, COUNT(*) AS cnt
            FROM article_feedback af
                        JOIN users u ON u.user_id = af.user_id AND NOT u.inactive
            WHERE (
                af.saved IS NOT NULL OR af.clicked_email IS NOT NULL OR af.clicked_web IS NOT NULL
            )
                            AND EXISTS (
                                SELECT 1 FROM user_publications up WHERE up.user_id = af.user_id
                            )
            GROUP BY af.user_id
            HAVING cnt >= %s
            ORDER BY af.user_id ASC
            """,
            (holdout_n + min_train,),
        )
        return [int(r[0]) for r in cur.fetchall()[: max(max_users, 0)]]


def _fetch_interactions(user_id: int) -> list[dict[str, Any]]:
    with closing(get_connection()) as conn, closing(conn.cursor(dictionary=True)) as cur:
        cur.execute(
            """
            SELECT
                af.article_id,
                a.title,
                a.abstract,
                (af.saved IS NOT NULL) AS has_saved,
                ((af.clicked_email IS NOT NULL) OR (af.clicked_web IS NOT NULL)) AS has_clicked,
                GREATEST(
                  IFNULL(af.saved, '1000-01-01'),
                  IFNULL(af.clicked_email, '1000-01-01'),
                                    IFNULL(af.clicked_web, '1000-01-01')
                ) AS event_time
            FROM article_feedback af
            JOIN articles a ON a.article_id = af.article_id
            WHERE af.user_id = %s
              AND (
                                af.saved IS NOT NULL OR af.clicked_email IS NOT NULL OR af.clicked_web IS NOT NULL
              )
            ORDER BY event_time ASC
            """,
            (user_id,),
        )
        rows = cur.fetchall() or []
    return rows


def _build_split(interactions: list[dict[str, Any]], holdout_n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], datetime]:
    if len(interactions) <= holdout_n:
        raise ValueError("Not enough interactions for holdout")
    train = interactions[:-holdout_n]
    holdout = interactions[-holdout_n:]
    split_ts = holdout[0]["event_time"]
    return train, holdout, split_ts


def _fetch_publications_before(user_id: int, split_ts: datetime, max_publications: int) -> list[dict[str, Any]]:
    with closing(get_connection()) as conn, closing(conn.cursor(dictionary=True)) as cur:
        cur.execute(
            """
            SELECT p.title, p.abstract, p.year, p.venue, p.citation_count, p.fields_of_study
            FROM user_publications up
            JOIN external_publications p ON p.publication_id = up.publication_id
            WHERE up.user_id = %s
              AND (up.created_at IS NULL OR up.created_at <= %s)
            ORDER BY COALESCE(p.year, 0) DESC, COALESCE(p.citation_count, 0) DESC, p.updated_at DESC
            LIMIT %s
            """,
            (user_id, split_ts, max_publications),
        )
        pubs = cur.fetchall() or []

        if not pubs:
            cur.execute(
                """
                SELECT p.title, p.abstract, p.year, p.venue, p.citation_count, p.fields_of_study
                FROM user_publications up
                JOIN external_publications p ON p.publication_id = up.publication_id
                WHERE up.user_id = %s
                ORDER BY COALESCE(p.year, 0) DESC, COALESCE(p.citation_count, 0) DESC, p.updated_at DESC
                LIMIT %s
                """,
                (user_id, max_publications),
            )
            pubs = cur.fetchall() or []

    out: list[dict[str, Any]] = []
    for p in pubs:
        fos = p.get("fields_of_study")
        try:
            fos = json.loads(fos) if isinstance(fos, str) else fos
        except Exception:
            pass
        out.append(
            {
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "citation_count": p.get("citation_count"),
                "fields_of_study": fos,
            }
        )
    return out


def _interaction_summary_from_train(train: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    sorted_train = sorted(
        train,
        key=lambda r: (
            int(bool(r.get("has_saved"))) * 5
            + int(bool(r.get("has_clicked"))) * 3,
            r.get("event_time"),
        ),
        reverse=True,
    )
    top = sorted_train[: max_examples]

    top_engaged = [
        {
            "title": r.get("title"),
            "abstract": r.get("abstract"),
            "has_saved": bool(r.get("has_saved")),
            "has_clicked": bool(r.get("has_clicked")),
            "event_time": str(r.get("event_time")),
        }
        for r in top
    ]

    return {
        "top_categories": [],
        "top_topics": [],
        "recent_interactions": [{"title": r.get("title"), "event_time": str(r.get("event_time"))} for r in top],
        "top_engaged_articles": top_engaged,
    }


def _build_holdout_items(holdout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in holdout:
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        abstract = str(row.get("abstract") or "").strip()
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        items.append(
            {
                "title": title,
                "abstract": abstract,
            }
        )
    return items


def _judge_prompt(user_id: int, variant_id: str, profile_text: str, holdout_items: list[dict[str, Any]]) -> str:
    holdout_lines: list[str] = []
    for i, item in enumerate(holdout_items, start=1):
        holdout_lines.append(f"{i}. title={item.get('title')}")
        if item.get("abstract"):
            holdout_lines.append(f"   abstract={item.get('abstract')}")
    holdout_block = "\n".join(holdout_lines)
    return (
        "Assess profile quality against held-out future interactions.\n\n"
        f"User ID: {user_id}\n"
        f"Variant: {variant_id}\n"
        f"Profile:\n{profile_text}\n\n"
        "Held-out interactions (title + abstract):\n"
        f"{holdout_block}\n\n"
        "Scoring rubric:\n"
        "Evaluate the profile using four dimensions: relevance, specificity, coverage, and consistency.\n"
        "For each dimension and overall, assign an INTEGER score from 1 to 5 only.\n"
        "Level meanings (must use exactly this scale):\n"
        "1 = very poor / fails criterion\n"
        "2 = weak / major issues\n"
        "3 = acceptable / mixed quality\n"
        "4 = strong / minor issues\n"
        "5 = excellent / fully satisfies criterion\n\n"
        "Dimensions:\n"
        "- relevance (1-5): alignment with held-out topics/tasks.\n"
        "- specificity (1-5): concrete technical detail; non-generic wording.\n"
        "- coverage (1-5): captures breadth of important held-out interests.\n"
        "- consistency (1-5): grammatical, semantic, and cross-sentence coherence.\n\n"
        "Calibration guidance:\n"
        "- Score 1-2 when criterion is clearly violated or mostly unsupported by evidence.\n"
        "- Score 3 for partial match with noticeable gaps.\n"
        "- Score 4 for strong match with minor issues.\n"
        "- Score 5 only for clear, specific, and well-supported excellence.\n\n"
        "Mandatory failure rules:\n"
        "1) If the text has malformed/clipped/ungrammatical sentences -> consistency=1 and overall=1 and pass_fail=fail.\n"
        "2) If the text is semantically vague or awkward (example style: 'I am interested in knowledge, with a focus on resource.') "
        "-> specificity=1 and consistency=1 and overall=1 and pass_fail=fail.\n"
        "3) If the text is generic boilerplate and could describe many unrelated users -> specificity<=2 and pass_fail=fail if severe.\n"
        "Return strict JSON with keys: relevance,specificity,coverage,consistency,overall,pass_fail,rationale_short. "
        "Scores are 1.0-5.0. pass_fail is pass/fail."
    )


def _pairwise_judge_prompt(
    user_id: int,
    variant_a: str,
    profile_a: str,
    variant_b: str,
    profile_b: str,
    holdout_items: list[dict[str, Any]],
) -> str:
    holdout_lines: list[str] = []
    for i, item in enumerate(holdout_items, start=1):
        holdout_lines.append(f"{i}. title={item.get('title')}")
        if item.get("abstract"):
            holdout_lines.append(f"   abstract={item.get('abstract')}")
    holdout_block = "\n".join(holdout_lines)

    return (
        "Compare two candidate user profiles against held-out future interactions.\n\n"
        f"User ID: {user_id}\n"
        f"Candidate A ({variant_a}):\n{profile_a}\n\n"
        f"Candidate B ({variant_b}):\n{profile_b}\n\n"
        "Held-out interactions (title + abstract):\n"
        f"{holdout_block}\n\n"
        "Use a pairwise rubric-based comparison protocol.\n"
        "Evaluate A and B independently against the same rubric before making a final decision.\n"
        "Choose the better profile using this priority: semantic quality/coherence first, then relevance, then specificity/coverage.\n"
        "To reduce position bias, treat A/B labels as arbitrary and base judgment only on content quality and evidence alignment.\n"
        "If one profile is awkward or semantically vague, it should lose.\n"
        "Return strict JSON with keys: winner,confidence,reason_short where winner in {A,B,tie} and confidence in [0.0,1.0]."
    )


def _winner_to_points(winner: str) -> tuple[float, float]:
    w = (winner or "").strip().lower()
    if w == "a":
        return 1.0, 0.0
    if w == "b":
        return 0.0, 1.0
    return 0.5, 0.5


def _save_split(experiment_id: str, user_id: int, split_ts: datetime) -> None:
    with closing(get_connection()) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            """
            INSERT INTO experiment_user_split(experiment_id, user_id, split_ts, train_start, train_end, holdout_start, holdout_end)
            VALUES (%s,%s,%s,NULL,%s,%s,NULL)
            ON DUPLICATE KEY UPDATE split_ts=VALUES(split_ts), train_end=VALUES(train_end), holdout_start=VALUES(holdout_start)
            """,
            (experiment_id, user_id, split_ts, split_ts, split_ts),
        )
        conn.commit()


def _clear_user_experiment_rows(experiment_id: str, user_id: int) -> None:
    with closing(get_connection()) as conn, closing(conn.cursor()) as cur:
        cur.execute("DELETE FROM experiment_topk_selection WHERE experiment_id=%s AND user_id=%s", (experiment_id, user_id))
        cur.execute("DELETE FROM experiment_judge_scores WHERE experiment_id=%s AND user_id=%s", (experiment_id, user_id))
        cur.execute("DELETE FROM experiment_profile_candidates WHERE experiment_id=%s AND user_id=%s", (experiment_id, user_id))
        conn.commit()


def _insert_candidate(experiment_id: str, user_id: int, variant_id: str, prompt_id: str, llm: str, profile_text: str | None, meta: dict[str, Any], status: str, error: str | None, run_id: str) -> None:
    with closing(get_connection()) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            """
            INSERT INTO experiment_profile_candidates(
                experiment_id,user_id,variant_id,prompt_id,llm,profile_text,meta_json,status,error_message,generated_at,run_id
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                prompt_id=VALUES(prompt_id), llm=VALUES(llm), profile_text=VALUES(profile_text),
                meta_json=VALUES(meta_json), status=VALUES(status), error_message=VALUES(error_message),
                generated_at=VALUES(generated_at), run_id=VALUES(run_id)
            """,
            (
                experiment_id,
                user_id,
                variant_id,
                prompt_id,
                llm,
                profile_text,
                json.dumps(meta, ensure_ascii=False) if meta else None,
                status,
                error,
                datetime.now(timezone.utc).replace(tzinfo=None),
                run_id,
            ),
        )
        conn.commit()


def _insert_judge_score(experiment_id: str, user_id: int, variant_id: str, judge_model: str, judge: dict[str, Any], holdout_items: list[dict[str, Any]]) -> None:
    with closing(get_connection()) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            """
            INSERT INTO experiment_judge_scores(
                experiment_id,user_id,variant_id,judge_model,relevance,specificity,coverage,consistency,overall,pass_fail,rationale_short,holdout_n,holdout_json
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                relevance=VALUES(relevance), specificity=VALUES(specificity), coverage=VALUES(coverage),
                consistency=VALUES(consistency), overall=VALUES(overall), pass_fail=VALUES(pass_fail),
                rationale_short=VALUES(rationale_short), holdout_n=VALUES(holdout_n), holdout_json=VALUES(holdout_json)
            """,
            (
                experiment_id,
                user_id,
                variant_id,
                judge_model,
                float(judge.get("relevance", 0)),
                float(judge.get("specificity", 0)),
                float(judge.get("coverage", 0)),
                float(judge.get("consistency", 0)),
                float(judge.get("overall", 0)),
                str(judge.get("pass_fail", "fail")).lower(),
                str(judge.get("rationale_short", ""))[:500],
                len(holdout_items),
                json.dumps(holdout_items, ensure_ascii=False),
            ),
        )
        conn.commit()


def _insert_topk(experiment_id: str, user_id: int, ranked: list[dict[str, Any]], top_k: int) -> None:
    with closing(get_connection()) as conn, closing(conn.cursor()) as cur:
        for idx, item in enumerate(ranked[:top_k], start=1):
            cur.execute(
                """
                INSERT INTO experiment_topk_selection(experiment_id,user_id,rank_pos,variant_id,prompt_id,selected_by,notes)
                VALUES (%s,%s,%s,%s,%s,'llm_judge',%s)
                ON DUPLICATE KEY UPDATE variant_id=VALUES(variant_id), prompt_id=VALUES(prompt_id), notes=VALUES(notes)
                """,
                (
                    experiment_id,
                    user_id,
                    idx,
                    item["variant_id"],
                    item["prompt_id"],
                    f"overall={item['judge'].get('overall')}; relevance={item['judge'].get('relevance')}",
                ),
            )
        conn.commit()


def main() -> None:
    load_project_env(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Run temporal split + 4-way generation + LLM judge.")
    parser.add_argument("--experiment-id", type=str, default=f"temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--users", type=str, default="")
    parser.add_argument("--max-users", type=int, default=5)
    parser.add_argument("--holdout-n", type=int, default=3)
    parser.add_argument("--min-train-interactions", type=int, default=5)
    parser.add_argument("--max-publications", type=int, default=30)
    parser.add_argument("--max-interaction-examples", type=int, default=30)
    parser.add_argument("--hybrid-weight-x", type=float, default=0.85)
    parser.add_argument(
        "--weighted-xs",
        type=str,
        default="0.2,0.5,0.8",
        help="Comma-separated publication-weight values for weighted variants in the experiment.",
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--disable-pairwise-ranking",
        action="store_true",
        help="Disable pairwise A/B judge comparisons and rank only by decimal pointwise rubric scores.",
    )
    parser.add_argument(
        "--include-mixed-variants",
        action="store_true",
        help="Include additional mixed variants that combine weighting and source-priority instruction.",
    )

    parser.add_argument("--gen-model", type=str, default="")
    parser.add_argument("--gen-api-key", type=str, default="")
    parser.add_argument("--gen-api-url", type=str, default="")
    parser.add_argument("--gen-base-url", type=str, default="")

    parser.add_argument("--judge-model", type=str, default="")
    parser.add_argument("--judge-api-base", type=str, default="")
    parser.add_argument("--judge-api-key", type=str, default="")

    args = parser.parse_args()

    users = _select_users(args.users, args.max_users, args.holdout_n, args.min_train_interactions)
    if not users:
        raise SystemExit("No eligible users")

    gen_model = pick_first_nonempty(
        args.gen_model,
        os.getenv("LLM_GEN_MODEL"),
        os.getenv("LLM_MODEL"),
        os.getenv("OLLAMA_MODEL"),
        "llama3.1:70b",
    )
    gen_api_url = pick_first_nonempty(
        args.gen_api_url,
        os.getenv("LLM_GEN_API_URL"),
        os.getenv("LLM_API_URL"),
        os.getenv("OLLAMA_API_URL"),
    )
    gen_base_url = pick_first_nonempty(
        args.gen_base_url,
        os.getenv("LLM_GEN_BASE_URL"),
        os.getenv("LLM_BASE_URL"),
        os.getenv("OLLAMA_HOST"),
    )
    gen_api_key = resolve_api_key(
        pick_first_nonempty(
            args.gen_api_key,
            os.getenv("LLM_GEN_API_KEY"),
            os.getenv("LLM_API_KEY"),
            os.getenv("OLLAMA_API_KEY"),
        ),
        endpoint_hint=f"{gen_api_url} {gen_base_url}",
    )

    judge_model = pick_first_nonempty(
        args.judge_model,
        os.getenv("LLM_JUDGE_MODEL"),
        os.getenv("JUDGE_MODEL"),
        os.getenv("LLM_MODEL"),
        "gpt-4o-mini",
    )
    judge_api_base = pick_first_nonempty(
        args.judge_api_base,
        os.getenv("LLM_JUDGE_API_BASE"),
        os.getenv("JUDGE_API_BASE"),
        os.getenv("LLM_API_BASE"),
        os.getenv("LLM_BASE_URL"),
        "https://api.openai.com/v1",
    )
    judge_api_key = resolve_api_key(
        pick_first_nonempty(
            args.judge_api_key,
            os.getenv("LLM_JUDGE_API_KEY"),
            os.getenv("JUDGE_API_KEY"),
            os.getenv("LLM_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ),
        endpoint_hint=judge_api_base,
    )

    if not gen_api_url and not gen_base_url:
        raise SystemExit("Missing generation endpoint (OLLAMA_HOST/--gen-base-url)")

    gen_client = LLMClient(
        LLMConfig(
            api_key=gen_api_key,
            model=gen_model,
            temperature=0.2,
            max_tokens=220,
            api_url=gen_api_url or None,
            base_url=gen_base_url or None,
            timeout=120,
            max_retries=6,
            retry_backoff_seconds=2.0,
        )
    )

    judge_client = LLMJudgeClient(
        LLMConnectionConfig(
            api_key=judge_api_key or None,
            api_base=judge_api_base,
            model=judge_model,
            temperature=0.0,
            timeout=90,
            max_retries=6,
            retry_backoff_seconds=2.0,
        ),
        system_prompt="You are an impartial critical evaluator. Return strict JSON only.",
    )
    pairwise_judge_client = LLMJudgeClient(
        LLMConnectionConfig(
            api_key=judge_api_key or None,
            api_base=judge_api_base,
            model=judge_model,
            temperature=0.0,
            timeout=90,
            max_retries=6,
            retry_backoff_seconds=2.0,
        ),
        system_prompt="You are an impartial critical evaluator. Return strict JSON only.",
        required_keys={"winner", "confidence", "reason_short"},
    )

    x = max(0.0, min(1.0, float(args.hybrid_weight_x)))
    weighted_x_values = _parse_weighted_x_values(args.weighted_xs)
    variants = _experiment_variants(weighted_x_values)

    if args.include_mixed_variants:
        variants.extend(
            [
                {
                    "variant_id": "mixed_publications_heavy",
                    "prompt_id": f"hybrid_mixed_pub_{x:.2f}",
                    "publication_weight": x,
                    "source_priority_instruction": "prioritize publication trajectory and long-term research themes; use interactions only as secondary calibration",
                    "prompt_variant": "mixed_publications_heavy",
                },
                {
                    "variant_id": "mixed_interactions_heavy",
                    "prompt_id": f"hybrid_mixed_int_{x:.2f}",
                    "publication_weight": 1.0 - x,
                    "source_priority_instruction": "prioritize interaction behavior and recent engagement; use publications only as background",
                    "prompt_variant": "mixed_interactions_heavy",
                },
            ]
        )

    run_id = str(uuid4())
    report: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "run_id": run_id,
        "users": users,
        "results": [],
    }

    for user_id in users:
        _clear_user_experiment_rows(args.experiment_id, user_id)
        interactions = _fetch_interactions(user_id)
        try:
            train, holdout, split_ts = _build_split(interactions, args.holdout_n)
        except Exception as exc:
            report["results"].append({"user_id": user_id, "status": "failed", "error": str(exc)})
            continue

        _save_split(args.experiment_id, user_id, split_ts)

        holdout_items = _build_holdout_items(holdout)
        holdout_titles = [str(item.get("title")) for item in holdout_items if item.get("title")]
        if not holdout_items:
            report["results"].append({"user_id": user_id, "status": "failed", "error": "empty holdout titles"})
            continue

        user_variants: list[dict[str, Any]] = []
        seen_profile_norm_to_variant: dict[str, str] = {}
        for v in variants:
            rec = {
                "variant_id": v["variant_id"],
                "prompt_id": v["prompt_id"],
                "status": "success",
            }
            try:
                pub_cap = args.max_publications
                int_cap = args.max_interaction_examples
                if v["publication_weight"] is not None:
                    w = _resolve_effective_weight(float(v["publication_weight"]))
                    pub_cap = max(1, min(args.max_publications, int(round(args.max_publications * w))))
                    int_cap = max(1, min(args.max_interaction_examples, int(round(args.max_interaction_examples * (1.0 - w)))))

                publications = _fetch_publications_before(user_id, split_ts, pub_cap)
                summary = _interaction_summary_from_train(train, int_cap)
                if not publications or not summary.get("top_engaged_articles"):
                    raise ValueError("missing train evidence")

                profile, meta = _generate_profile_from_hybrid_evidence(
                    gen_client,
                    user_id,
                    publications,
                    summary,
                    publication_weight=v["publication_weight"],
                    source_priority_instruction=v["source_priority_instruction"],
                    prompt_variant=v["prompt_variant"],
                    weighted_publication_cap=pub_cap,
                    weighted_interaction_cap=int_cap,
                )

                profile_norm = _normalize_profile_for_dedup(profile)
                duplicate_of = seen_profile_norm_to_variant.get(profile_norm)
                if duplicate_of:
                    raise ValueError(f"duplicate_profile_of:{duplicate_of}")
                seen_profile_norm_to_variant[profile_norm] = v["variant_id"]

                meta["split_ts"] = str(split_ts)
                meta["train_interactions"] = len(train)
                meta["holdout_interactions"] = len(holdout)
                rec["profile_text"] = profile
                rec["meta"] = meta
                _insert_candidate(
                    args.experiment_id,
                    user_id,
                    v["variant_id"],
                    v["prompt_id"],
                    args.gen_model,
                    profile,
                    meta,
                    "success",
                    None,
                    run_id,
                )
            except Exception as exc:
                rec["status"] = "failed"
                rec["error"] = str(exc)
                _insert_candidate(
                    args.experiment_id,
                    user_id,
                    v["variant_id"],
                    v["prompt_id"],
                    args.gen_model,
                    None,
                    {},
                    "failed",
                    str(exc),
                    run_id,
                )

            user_variants.append(rec)

        scored: list[dict[str, Any]] = []
        for rec in user_variants:
            if rec.get("status") != "success":
                continue
            try:
                judge = judge_client.evaluate(
                    _judge_prompt(user_id, rec["variant_id"], rec["profile_text"], holdout_items)
                )
                judge = _apply_judge_quality_guard(judge, rec["profile_text"])
                rec["judge"] = judge
                scored.append(rec)
                _insert_judge_score(
                    args.experiment_id,
                    user_id,
                    rec["variant_id"],
                    args.judge_model,
                    judge,
                    holdout_items,
                )
            except Exception as exc:
                rec["judge_error"] = str(exc)

        passed = [r for r in scored if str(r.get("judge", {}).get("pass_fail", "")).lower() == "pass"]
        ranking_pool = passed if passed else scored

        pairwise_matches: list[dict[str, Any]] = []
        for rec in ranking_pool:
            rec["pairwise_points"] = 0.0

        if not args.disable_pairwise_ranking:
            for a_idx, b_idx in combinations(range(len(ranking_pool)), 2):
                a = ranking_pool[a_idx]
                b = ranking_pool[b_idx]
                try:
                    comp = pairwise_judge_client.evaluate(
                        _pairwise_judge_prompt(
                            user_id=user_id,
                            variant_a=str(a.get("variant_id")),
                            profile_a=str(a.get("profile_text") or ""),
                            variant_b=str(b.get("variant_id")),
                            profile_b=str(b.get("profile_text") or ""),
                            holdout_items=holdout_items,
                        )
                    )
                    pa, pb = _winner_to_points(str(comp.get("winner", "tie")))
                    conf = float(comp.get("confidence", 0.5))
                    conf = max(0.0, min(1.0, conf))
                    # Confidence-weighted points to reduce random ties.
                    a["pairwise_points"] = float(a.get("pairwise_points", 0.0)) + pa * (0.5 + 0.5 * conf)
                    b["pairwise_points"] = float(b.get("pairwise_points", 0.0)) + pb * (0.5 + 0.5 * conf)
                    pairwise_matches.append(
                        {
                            "a": a.get("variant_id"),
                            "b": b.get("variant_id"),
                            "winner": comp.get("winner", "tie"),
                            "confidence": conf,
                            "reason_short": str(comp.get("reason_short", ""))[:240],
                        }
                    )
                except Exception as exc:
                    pairwise_matches.append(
                        {
                            "a": a.get("variant_id"),
                            "b": b.get("variant_id"),
                            "winner": "tie",
                            "confidence": 0.0,
                            "reason_short": f"pairwise_error:{exc}",
                        }
                    )

        ranked = sorted(
            ranking_pool,
            key=lambda r: (
                float(r.get("pairwise_points", 0.0)),
                float(r["judge"].get("overall", 0)),
                float(r["judge"].get("consistency", 0)),
                float(r["judge"].get("relevance", 0)),
                float(r["judge"].get("specificity", 0)),
                float(r["judge"].get("coverage", 0)),
            ),
            reverse=True,
        )
        _insert_topk(args.experiment_id, user_id, ranked, max(args.top_k, 1))

        report["results"].append(
            {
                "user_id": user_id,
                "status": "success",
                "split_ts": str(split_ts),
                "holdout_titles": holdout_titles,
                "holdout_items": holdout_items,
                "generated_variants": [
                    {
                        "variant_id": r.get("variant_id"),
                        "prompt_id": r.get("prompt_id"),
                        "status": r.get("status"),
                        "profile_text": r.get("profile_text"),
                        "error": r.get("error"),
                        "judge": r.get("judge"),
                        "pairwise_points": r.get("pairwise_points", 0.0),
                        "judge_error": r.get("judge_error"),
                    }
                    for r in user_variants
                ],
                "pairwise_matches": pairwise_matches,
                "top_selected": [
                    {
                        "variant_id": r["variant_id"],
                        "prompt_id": r["prompt_id"],
                        "overall": r["judge"].get("overall"),
                        "pairwise_points": r.get("pairwise_points", 0.0),
                    }
                    for r in ranked[: max(args.top_k, 1)]
                ],
            }
        )

    out_path = PROJECT_ROOT / "scripts" / "data" / f"temporal_experiment_{args.experiment_id}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {out_path}")
    for row in report["results"]:
        if row.get("status") == "success":
            print(f"user {row['user_id']}: top -> {[x['variant_id'] for x in row.get('top_selected', [])]}")
        else:
            print(f"user {row['user_id']}: failed ({row.get('error')})")


if __name__ == "__main__":
    main()
