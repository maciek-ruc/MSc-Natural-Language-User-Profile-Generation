"""Minimal single-prompt NL profile generator for Ollama-compatible endpoints.

Design goals:
- one shared system prompt
- one shared user prompt
- structured JSON evidence only
- no rewrite policy
- no DB persistence side effects
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arxivdigest.core.database.connection import get_connection
from scripts.llm_connector import LLMConnectionConfig
from scripts.llm_connector import OpenAICompatibleLLMClient
from scripts.llm_connector import load_project_env
from scripts.llm_connector import resolve_api_key


DEFAULT_MODEL = "gorina10.llama3.3:70b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 220
DEFAULT_SYSTEM_PROMPT = (
    "You write concise first-person research-interest profiles for recommendation systems. "
    "Return only the final profile as plain text. Start with 'I' or 'My'. "
    "Do not use headings, bullets, labels, or markdown."
)
STYLE_EXAMPLE = (
    "I am interested in a coherent mix of long-term research themes and recent directions reflected in my activity. "
    "I focus on concrete methods, domains, and evaluation questions rather than generic topic labels. "
    "My profile should read like a compact description of sustained research interests."
)
MISSING_ABSTRACT_PLACEHOLDER = "[ABSTRACT_MISSING_IN_DB]"


def _resolve_effective_weight(weight: float | None, default: float = 0.5) -> float:
    if weight is None:
        return float(default)
    w = max(0.0, min(1.0, float(weight)))
    if w == 0.0:
        return float(default)
    return w


def _simple_variant_specs(x: float) -> dict[str, dict[str, Any]]:
    weight_x = _resolve_effective_weight(x)
    balanced_weight = _resolve_effective_weight(0.5)
    return {
        "weighted": {
            "variant_id": "weighted",
            "prompt_id": f"hybrid_weighted_{weight_x:.2f}",
            "publication_weight": weight_x,
            "source_priority_instruction": None,
            "prompt_variant": "weighted",
        },
        "prompt_publications_priority": {
            "variant_id": "prompt_publications_priority",
            "prompt_id": "hybrid_prompt_pub_priority",
            "publication_weight": balanced_weight,
            "source_priority_instruction": "focus mainly on publication evidence when selecting the main themes; use interactions as secondary support",
            "prompt_variant": "prompt_publications_priority",
        },
        "prompt_interactions_priority": {
            "variant_id": "prompt_interactions_priority",
            "prompt_id": "hybrid_prompt_interactions_priority",
            "publication_weight": balanced_weight,
            "source_priority_instruction": "focus mainly on interaction evidence when selecting the main themes; use publications as background support",
            "prompt_variant": "prompt_interactions_priority",
        },
    }


def _simple_experiment_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for value in [0.2, 0.5, 0.8]:
        spec = dict(_simple_variant_specs(value)["weighted"])
        suffix = f"{value:.2f}"
        spec["variant_id"] = f"weighted_x_{suffix}"
        spec["prompt_variant"] = f"weighted_x_{suffix}"
        variants.append(spec)

    balanced_priority_specs = _simple_variant_specs(0.5)
    variants.append(dict(balanced_priority_specs["prompt_publications_priority"]))
    variants.append(dict(balanced_priority_specs["prompt_interactions_priority"]))
    return variants


def _resolve_prompt_priority(source_priority_instruction: str | None) -> str:
    if not source_priority_instruction:
        return "balanced"
    lower_rule = source_priority_instruction.lower()
    if "focus mainly on publication evidence" in lower_rule:
        return "publications"
    if "focus mainly on interaction evidence" in lower_rule:
        return "interactions"
    return "balanced"


@dataclass
class LLMConfig(LLMConnectionConfig):
    max_tokens: int = DEFAULT_MAX_TOKENS


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAICompatibleLLMClient(config)

    def generate(self, prompt: str, system_prompt: str) -> str:
        return self._client.chat_text(
            prompt,
            system_prompt,
            max_tokens=self.config.max_tokens,
        ).strip()


def _safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def _select_user_ids(limit_users: int | None) -> list[int]:
    with closing(get_connection()) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            """
            SELECT DISTINCT u.user_id
            FROM users u
            WHERE NOT u.inactive
              AND EXISTS (SELECT 1 FROM user_publications up WHERE up.user_id = u.user_id)
              AND (
                    EXISTS (
                        SELECT 1
                        FROM article_feedback af
                        WHERE af.user_id = u.user_id
                          AND (
                                af.clicked_email IS NOT NULL OR af.clicked_web IS NOT NULL
                                OR af.saved IS NOT NULL
                              )
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM topic_recommendations tr
                        WHERE tr.user_id = u.user_id
                          AND tr.clicked IS NOT NULL
                    )
              )
            ORDER BY u.user_id ASC
            """
        )
        user_ids = [int(row[0]) for row in cur.fetchall()]

    if limit_users and limit_users > 0:
        return user_ids[:limit_users]
    return user_ids


def _get_user_publications(user_id: int, max_publications: int) -> list[dict[str, Any]]:
    with closing(get_connection()) as conn, closing(conn.cursor(dictionary=True)) as cur:
        cur.execute(
            """
            SELECT
                p.title,
                p.abstract,
                p.year,
                p.venue,
                p.citation_count,
                p.fields_of_study,
                p.publication_types
            FROM user_publications up
            JOIN external_publications p ON p.publication_id = up.publication_id
            WHERE up.user_id = %s
            ORDER BY COALESCE(p.year, 0) DESC, COALESCE(p.citation_count, 0) DESC, p.updated_at DESC
            LIMIT %s
            """,
            (user_id, max_publications),
        )
        rows = cur.fetchall() or []

    return [
        {
            "title": row.get("title"),
            "abstract": row.get("abstract"),
            "year": row.get("year"),
            "venue": row.get("venue"),
            "citation_count": row.get("citation_count"),
            "fields_of_study": _safe_json_loads(row.get("fields_of_study")),
            "publication_types": _safe_json_loads(row.get("publication_types")),
        }
        for row in rows
    ]


def _get_user_interaction_summary(user_id: int, max_examples: int) -> dict[str, Any]:
    with closing(get_connection()) as conn, closing(conn.cursor(dictionary=True)) as cur:
        cur.execute(
            """
                        SELECT t.topic, ut.interaction_time
                        FROM user_topics ut
                        JOIN topics t ON t.topic_id = ut.topic_id
                        WHERE ut.user_id = %s
                            AND ut.state IN ('USER_ADDED', 'SYSTEM_RECOMMENDED_ACCEPTED')
                            AND NOT t.filtered
                        ORDER BY ut.interaction_time DESC, t.topic ASC
            LIMIT 10
            """,
            (user_id,),
        )
        top_topics = cur.fetchall() or []

        cur.execute(
            """
            SELECT
                a.title,
                a.abstract,
                GREATEST(
                  IFNULL(af.saved, '1000-01-01'),
                  IFNULL(af.clicked_email, '1000-01-01'),
                  IFNULL(af.clicked_web, '1000-01-01')
                ) AS event_time,
                a.article_id
            FROM article_feedback af
            JOIN articles a ON a.article_id = af.article_id
            WHERE af.user_id = %s
              AND (
                    af.saved IS NOT NULL OR af.clicked_email IS NOT NULL OR af.clicked_web IS NOT NULL
              )
            ORDER BY event_time DESC
            LIMIT %s
            """,
            (user_id, max_examples),
        )
        interacted_articles = cur.fetchall() or []

    return {
        "top_topics": [{"topic": row.get("topic")} for row in top_topics],
        "interacted_articles": [
            {
                "article_id": row.get("article_id"),
                "title": row.get("title"),
                "abstract": row.get("abstract"),
                "event_time": str(row.get("event_time")),
            }
            for row in interacted_articles
        ],
    }


def _has_sufficient_interaction_data(summary: dict[str, Any], min_interacted_articles: int = 1) -> bool:
    return len(summary.get("interacted_articles", [])) >= max(1, int(min_interacted_articles))


def _truncate_abstract(text: Any, max_chars: int) -> str:
    abstract = str(text or "").strip()
    if len(abstract) > max_chars:
        abstract = abstract[:max_chars] + "..."
    return abstract or MISSING_ABSTRACT_PLACEHOLDER


def _build_evidence_json(publications: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "publications": [
            {
                "title": pub.get("title"),
                "abstract": _truncate_abstract(pub.get("abstract"), 400),
                "year": pub.get("year"),
                "venue": pub.get("venue"),
                "citation_count": pub.get("citation_count"),
                "fields_of_study": pub.get("fields_of_study"),
                "publication_types": pub.get("publication_types"),
            }
            for pub in publications
        ],
        "interaction_topics": summary.get("top_topics", []),
        "interactions": [
            {
                "article_id": item.get("article_id"),
                "title": item.get("title"),
                "abstract": _truncate_abstract(item.get("abstract"), 350),
                "event_time": item.get("event_time"),
            }
            for item in summary.get("interacted_articles", [])
        ],
    }


def _build_prompt(
    user_id: int,
    publications: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    source_priority_instruction: str | None = None,
) -> str:
    evidence_json = _build_evidence_json(publications, summary)
    evidence_json_text = json.dumps(evidence_json, ensure_ascii=False, indent=2)
    priority = _resolve_prompt_priority(source_priority_instruction)
    objective_line = "Use publication history and interaction behavior together to form one coherent profile."
    if priority == "publications":
        objective_line = "Use both evidence sources, but prioritize publication evidence when selecting the main themes."
    elif priority == "interactions":
        objective_line = "Use both evidence sources, but prioritize interaction evidence when selecting the main themes."
    return "\n".join(
        [
            f"User ID: {user_id}",
            "Write one concise first-person research-interest profile in English.",
            "Return plain text only, 3 to 4 sentences.",
            objective_line,
            "Prefer specific themes, methods, domains, and evaluation concerns over broad umbrella labels.",
            "Avoid generic phrases unless they are strongly supported by the evidence.",
            "Interpret the JSON only as structured evidence about the user's research interests and activities.",
            "Use only the content values in the JSON to infer themes, methods, tasks, domains, and evaluation concerns.",
            "Do not describe the JSON structure, keys, arrays, field names, counts, nulls, or missing sections.",
            "Do not write phrases such as 'based on the provided text', 'the JSON shows', 'this section is empty', 'there are N items', or 'the conversation appears to be'.",
            "Do not list topics explicitly or explain the evidence; write the final profile itself.",
            "Do not mention the prompt, the data, uncertainty, missing information, or formatting instructions.",
            "Do not use headings, bullets, labels, or meta-commentary.",
            "Start directly with 'I' or 'My'.",
            "Style example:",
            STYLE_EXAMPLE,
            "Do not copy the example; infer the actual interests from the evidence below.",
            *(
                [f"Priority rule: {source_priority_instruction}"]
                if source_priority_instruction
                else []
            ),
            "",
            "Evidence JSON:",
            evidence_json_text,
        ]
    )


def _sanitize_profile(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+[.)]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if len(sentences) > 4:
        sentences = sentences[:4]
    cleaned = " ".join(sentences).strip()
    lower = cleaned.lower()
    if cleaned and not (lower.startswith("i ") or lower.startswith("i am") or lower.startswith("i'm") or lower.startswith("my ")):
        cleaned = "I am interested in " + cleaned[0].lower() + cleaned[1:]
    return cleaned


def _generate_profile(
    client: LLMClient,
    user_id: int,
    max_publications: int,
    max_examples: int,
    min_interacted_articles: int,
    *,
    publication_weight: float | None = None,
    source_priority_instruction: str | None = None,
    prompt_variant: str = "default",
) -> tuple[str, dict[str, Any]]:
    weighted_publication_cap = max_publications
    weighted_interaction_cap = max_examples
    if publication_weight is not None:
        publication_weight = _resolve_effective_weight(publication_weight)
        weighted_publication_cap = max(1, min(max_publications, int(round(max_publications * publication_weight))))
        weighted_interaction_cap = max(1, min(max_examples, int(round(max_examples * (1.0 - publication_weight)))))

    publications = _get_user_publications(user_id, weighted_publication_cap)
    summary = _get_user_interaction_summary(user_id, weighted_interaction_cap)
    if not publications:
        raise ValueError("skipped_insufficient_publication_data")
    if not _has_sufficient_interaction_data(summary, min_interacted_articles=min_interacted_articles):
        raise ValueError(f"skipped_insufficient_interaction_data:min_interacted_articles={min_interacted_articles}")

    prompt = _build_prompt(
        user_id,
        publications,
        summary,
        source_priority_instruction=source_priority_instruction,
    )
    profile_text = _sanitize_profile(client.generate(prompt, DEFAULT_SYSTEM_PROMPT))
    return profile_text, {
        "prompt_variant": prompt_variant,
        "publication_weight": publication_weight,
        "weighted_publication_cap": weighted_publication_cap,
        "weighted_interaction_cap": weighted_interaction_cap,
        "source_priority_instruction": source_priority_instruction,
        "publication_count": len(publications),
        "interaction_count": len(summary.get("interacted_articles", [])),
        "evidence_json": _build_evidence_json(publications, summary),
        "recent_publications": [
            {"title": pub.get("title"), "year": pub.get("year"), "venue": pub.get("venue")}
            for pub in publications[:10]
        ],
        "interaction_summary": summary,
        "prompt_preview": prompt[:2000],
    }


def _resolve_output_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_output_snapshot(output_path: Path, output: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    load_project_env(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Generate concise NL user profiles with one Ollama-oriented prompt.")
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--user-ids-file", type=str, default="")
    parser.add_argument("--limit-users", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--batch-index", type=int, default=1)
    parser.add_argument("--max-publications", type=int, default=30)
    parser.add_argument("--max-interaction-examples", type=int, default=30)
    parser.add_argument("--min-interacted-articles", type=int, default=1)
    parser.add_argument(
        "--run-hybrid-5way",
        action="store_true",
        help="Run the five experiment variants: x=0.20, x=0.50, x=0.80, publications-priority, interactions-priority.",
    )
    parser.add_argument(
        "--run-hybrid-4way",
        action="store_true",
        help="Backward-compatible alias for --run-hybrid-5way.",
    )
    parser.add_argument("--hybrid-weight-x", type=float, default=0.8)
    parser.add_argument("--publication-weight", type=float, default=0.5)
    parser.add_argument(
        "--variant-id",
        type=str,
        default="weighted",
        choices=["weighted", "prompt_publications_priority", "prompt_interactions_priority"],
    )
    parser.add_argument("--output", type=str, default="scripts/data/user_nl_profiles_ollama_simple.json")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--llm-api-key", type=str, default="")
    parser.add_argument("--llm-api-url", type=str, default="")
    parser.add_argument("--llm-base-url", type=str, default="")
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("LLM_TIMEOUT", "60")))
    args = parser.parse_args()

    if args.user_id is not None and args.user_ids_file.strip():
        raise ValueError("Use either --user-id or --user-ids-file, not both.")
    if args.batch_size < 0:
        raise ValueError("--batch-size must be >= 0.")
    if args.batch_index < 1:
        raise ValueError("--batch-index must be >= 1.")
    if not (0.0 <= float(args.publication_weight) <= 1.0):
        raise ValueError("--publication-weight must be in [0,1].")

    model = args.model.strip() or DEFAULT_MODEL
    api_url = args.llm_api_url.strip() or os.getenv("OLLAMA_API_URL") or os.getenv("LLM_API_URL")
    base_url = args.llm_base_url.strip() or os.getenv("OLLAMA_HOST") or os.getenv("LLM_BASE_URL")
    api_key = resolve_api_key(args.llm_api_key, endpoint_hint=f"{api_url} {base_url}")

    if not api_url and not base_url:
        raise ValueError("LLM endpoint is required. Set OLLAMA_HOST/OLLAMA_API_URL or pass --llm-base-url/--llm-api-url.")

    client = LLMClient(
        LLMConfig(
            api_key=api_key,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_url=api_url or None,
            base_url=base_url or None,
            timeout=args.timeout,
        )
    )

    if args.user_ids_file.strip():
        user_ids_path = Path(args.user_ids_file).expanduser()
        if not user_ids_path.is_absolute():
            user_ids_path = (PROJECT_ROOT / user_ids_path).resolve()
        loaded_user_ids = json.loads(user_ids_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_user_ids, list) or not all(isinstance(value, int) for value in loaded_user_ids):
            raise ValueError("--user-ids-file must point to a JSON array of integers.")
        user_ids = loaded_user_ids
    elif args.user_id is not None:
        user_ids = [args.user_id]
    else:
        user_ids = _select_user_ids(args.limit_users)

    run_hybrid_5way = bool(args.run_hybrid_5way or args.run_hybrid_4way)

    if run_hybrid_5way and args.user_id is None:
        raise ValueError("--run-hybrid-5way requires --user-id (single-user run).")

    if args.batch_size:
        start = (args.batch_index - 1) * args.batch_size
        end = start + args.batch_size
        if start >= len(user_ids):
            raise ValueError(
                f"--batch-index={args.batch_index} is out of range for {len(user_ids)} users and batch-size={args.batch_size}."
            )
        user_ids = user_ids[start:end]

    output_path = _resolve_output_path(args.output)
    generated_at_dt = datetime.now(timezone.utc)
    output = {
        "generated_at": generated_at_dt.isoformat(),
        "run_id": str(uuid4()),
        "run_hybrid_5way": run_hybrid_5way,
        "run_hybrid_4way": bool(args.run_hybrid_4way),
        "hybrid_weight_x": float(args.hybrid_weight_x),
        "publication_weight": float(args.publication_weight),
        "variant_id": args.variant_id,
        "model": args.model,
        "resolved_model": model,
        "user_count": len(user_ids),
        "records": [],
    }
    _write_output_snapshot(output_path, output)

    variant_specs = _simple_variant_specs(float(args.hybrid_weight_x))

    total_users = len(user_ids)
    for index, user_id in enumerate(user_ids, start=1):
        print(f"[{index}/{total_users}] user_id={user_id}", flush=True)
        user_records: list[dict[str, Any]] = []

        if run_hybrid_5way:
            selected_variants = _simple_experiment_variants()
        else:
            selected_variant = variant_specs[args.variant_id]
            if args.variant_id == "weighted":
                selected_variant = dict(selected_variant)
                selected_variant["publication_weight"] = _resolve_effective_weight(float(args.publication_weight))
            selected_variants = [selected_variant]

        for variant in selected_variants:
            record: dict[str, Any] = {
                "user_id": user_id,
                "status": "success",
                "variant_id": variant["variant_id"],
                "prompt_id": variant["prompt_id"],
                "llm": model,
            }
            try:
                profile_text, meta = _generate_profile(
                    client,
                    user_id,
                    args.max_publications,
                    args.max_interaction_examples,
                    args.min_interacted_articles,
                    publication_weight=variant.get("publication_weight"),
                    source_priority_instruction=variant.get("source_priority_instruction"),
                    prompt_variant=str(variant.get("prompt_variant") or variant.get("variant_id") or "default"),
                )
                record["profile_text"] = profile_text
                record["meta"] = meta
            except Exception as exc:
                error_text = str(exc)
                record["status"] = "skipped" if error_text.startswith("skipped_") else "failed"
                record["error"] = error_text
            user_records.append(record)

        output["records"].extend(user_records)
        _write_output_snapshot(output_path, output)

    print(f"Saved profiles for {len(user_ids)} users to {output_path}")


if __name__ == "__main__":
    main()