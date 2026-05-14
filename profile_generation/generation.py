from __future__ import annotations

from contextlib import closing
import json
import re
from typing import Any

from profile_generation.db import get_connection
from profile_generation.llm import LLMClient


MISSING_ABSTRACT_PLACEHOLDER = "[ABSTRACT_MISSING_IN_DB]"


def resolve_effective_weight(weight: float | None, default: float = 0.5) -> float:
    if weight is None:
        return float(default)
    value = max(0.0, min(1.0, float(weight)))
    return float(default) if value == 0.0 else value


def experiment_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for value in [0.2, 0.5, 0.8]:
        suffix = f"{value:.2f}"
        variants.append(
            {
                "variant_id": f"weighted_x_{suffix}",
                "prompt_id": f"hybrid_weighted_{suffix}",
                "publication_weight": value,
                "source_priority_instruction": None,
                "prompt_variant": f"weighted_x_{suffix}",
            }
        )
    variants.append(
        {
            "variant_id": "prompt_publications_priority",
            "prompt_id": "hybrid_prompt_pub_priority",
            "publication_weight": 0.5,
            "source_priority_instruction": "focus mainly on publication evidence when selecting the main themes; use interactions as secondary support",
            "prompt_variant": "prompt_publications_priority",
        }
    )
    variants.append(
        {
            "variant_id": "prompt_interactions_priority",
            "prompt_id": "hybrid_prompt_interactions_priority",
            "publication_weight": 0.5,
            "source_priority_instruction": "focus mainly on interaction evidence when selecting the main themes; use publications as background support",
            "prompt_variant": "prompt_interactions_priority",
        }
    )
    return variants


def _safe_json_loads(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def get_user_publications(user_id: int, max_publications: int) -> list[dict[str, Any]]:
    with closing(get_connection()) as conn, closing(conn.cursor(dictionary=True)) as cur:
        cur.execute(
            """
            SELECT p.title, p.abstract, p.year, p.venue, p.citation_count, p.fields_of_study, p.publication_types
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


def get_user_interactions(user_id: int, max_examples: int) -> dict[str, Any]:
    with closing(get_connection()) as conn, closing(conn.cursor(dictionary=True)) as cur:
        cur.execute(
            """
            SELECT a.article_id, a.title, a.abstract,
                   GREATEST(IFNULL(af.saved, '1000-01-01'), IFNULL(af.clicked_email, '1000-01-01'), IFNULL(af.clicked_web, '1000-01-01')) AS event_time
            FROM article_feedback af
            JOIN articles a ON a.article_id = af.article_id
            WHERE af.user_id = %s
              AND (af.saved IS NOT NULL OR af.clicked_email IS NOT NULL OR af.clicked_web IS NOT NULL)
            ORDER BY event_time DESC
            LIMIT %s
            """,
            (user_id, max_examples),
        )
        rows = cur.fetchall() or []
    return {
        "interacted_articles": [
            {
                "article_id": row.get("article_id"),
                "title": row.get("title"),
                "abstract": row.get("abstract"),
                "event_time": str(row.get("event_time")),
            }
            for row in rows
        ]
    }


def truncate_abstract(text: Any, max_chars: int) -> str:
    abstract = str(text or "").strip()
    if len(abstract) > max_chars:
        abstract = abstract[:max_chars] + "..."
    return abstract or MISSING_ABSTRACT_PLACEHOLDER


def build_evidence_json(publications: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "publications": [
            {"title": pub.get("title"), "abstract": truncate_abstract(pub.get("abstract"), 400)}
            for pub in publications
        ],
        "interactions": [
            {"title": item.get("title"), "abstract": truncate_abstract(item.get("abstract"), 350)}
            for item in summary.get("interacted_articles", [])
        ],
    }


def resolve_realized_source_counts(
    available_publications: int,
    available_interactions: int,
    publication_weight: float | None,
    *,
    max_total_items: int,
    min_interactions: int = 1,
) -> tuple[int, int]:
    total_available = max(available_publications, 0) + max(available_interactions, 0)
    total_cap = min(max(0, int(max_total_items)), total_available)
    if available_publications <= 0 or available_interactions <= 0:
        return min(max(available_publications, 0), total_cap), min(max(available_interactions, 0), total_cap)

    weight = resolve_effective_weight(publication_weight)
    required_interactions = min(max(1, int(min_interactions)), available_interactions)
    total_cap = min(max(total_cap, required_interactions + 1), total_available)

    best_publications = 1
    best_interactions = required_interactions
    best_error = float("inf")
    best_total = -1
    max_publications = min(available_publications, total_cap - required_interactions)
    for publication_count in range(1, max_publications + 1):
        max_interactions = min(available_interactions, total_cap - publication_count)
        for interaction_count in range(required_interactions, max_interactions + 1):
            total = publication_count + interaction_count
            error = abs((publication_count / total) - weight)
            if error < best_error or (error == best_error and total > best_total):
                best_publications = publication_count
                best_interactions = interaction_count
                best_error = error
                best_total = total
    return best_publications, best_interactions


def select_weighted_evidence(
    publications: list[dict[str, Any]],
    summary: dict[str, Any],
    publication_weight: float | None,
    *,
    max_total_items: int,
    min_interactions: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
    interaction_items = list(summary.get("interacted_articles", []))
    publication_count, interaction_count = resolve_realized_source_counts(
        len(publications),
        len(interaction_items),
        publication_weight,
        max_total_items=max_total_items,
        min_interactions=min_interactions,
    )
    return (
        publications[:publication_count],
        {"interacted_articles": interaction_items[:interaction_count]},
        publication_count,
        interaction_count,
    )


def select_fixed_evidence(
    publications: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    publication_cap: int,
    interaction_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
    selected_publications = list(publications[: max(0, int(publication_cap))])
    selected_interactions = list(summary.get("interacted_articles", [])[: max(0, int(interaction_cap))])
    return (
        selected_publications,
        {"interacted_articles": selected_interactions},
        len(selected_publications),
        len(selected_interactions),
    )


def build_prompt(
    user_id: int,
    publications: list[dict[str, Any]],
    summary: dict[str, Any],
    source_priority_instruction: str | None = None,
) -> str:
    objective_line = "Use publication history and interaction behavior together to form one coherent profile."
    if source_priority_instruction and "publication evidence" in source_priority_instruction.lower():
        objective_line = "Use both evidence sources, but prioritize publication evidence when selecting the main themes."
    elif source_priority_instruction and "interaction evidence" in source_priority_instruction.lower():
        objective_line = "Use both evidence sources, but prioritize interaction evidence when selecting the main themes."
    evidence_json_text = json.dumps(build_evidence_json(publications, summary), ensure_ascii=False, indent=2)
    lines = [
        f"User ID: {user_id}",
        "Write one concise first-person research-interest profile in English.",
        "Return plain text only, 3 to 4 sentences.",
        objective_line,
        "Prefer specific themes, methods, domains, and evaluation concerns over broad umbrella labels.",
        "Infer the actual interests from the evidence below.",
    ]
    if source_priority_instruction:
        lines.append(f"Priority rule: {source_priority_instruction}")
    lines.extend(["", "Evidence JSON:", evidence_json_text])
    return "\n".join(lines)


def sanitize_profile(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+[.)]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    cleaned = " ".join(sentences[:4]).strip()
    lower = cleaned.lower()
    if cleaned and not (lower.startswith("i ") or lower.startswith("i am") or lower.startswith("i'm") or lower.startswith("my ")):
        cleaned = "I am interested in " + cleaned[0].lower() + cleaned[1:]
    return cleaned


def generate_profile_text_from_evidence(
    client: LLMClient,
    user_id: int,
    variant: dict[str, Any],
    publications: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    shared_evidence_cap: int = 30,
    min_interacted_articles: int = 1,
    fixed_priority_source_cap: int = 15,
    max_tokens: int = 220,
) -> dict[str, Any]:
    if not publications:
        raise ValueError("skipped_insufficient_publication_data")
    if len(summary.get("interacted_articles", [])) < max(1, int(min_interacted_articles)):
        raise ValueError("skipped_insufficient_interaction_data")

    use_fixed = str(variant.get("variant_id") or "") in {"prompt_publications_priority", "prompt_interactions_priority"}
    if use_fixed:
        selected_publications, selected_summary, pub_cap, int_cap = select_fixed_evidence(
            publications,
            summary,
            publication_cap=fixed_priority_source_cap,
            interaction_cap=fixed_priority_source_cap,
        )
        mode = "priority_fixed_cap"
    else:
        selected_publications, selected_summary, pub_cap, int_cap = select_weighted_evidence(
            publications,
            summary,
            variant.get("publication_weight"),
            max_total_items=shared_evidence_cap,
            min_interactions=min_interacted_articles,
        )
        mode = "weighted_shared_cap"

    prompt = build_prompt(user_id, selected_publications, selected_summary, variant.get("source_priority_instruction"))
    profile_text = sanitize_profile(client.chat_text(prompt, max_tokens=max_tokens))
    return {
        "user_id": user_id,
        "variant_id": variant["variant_id"],
        "prompt_id": variant["prompt_id"],
        "profile_text": profile_text,
        "status": "success",
        "meta": {
            "publication_count": len(selected_publications),
            "interaction_count": len(selected_summary.get("interacted_articles", [])),
            "weighted_publication_cap": pub_cap,
            "weighted_interaction_cap": int_cap,
            "source_priority_instruction": variant.get("source_priority_instruction"),
            "publication_weight": variant.get("publication_weight"),
            "prompt_variant": variant.get("prompt_variant"),
            "evidence_selection_mode": mode,
            "evidence_json": build_evidence_json(selected_publications, selected_summary),
            "prompt_preview": prompt[:2000],
        },
    }


def generate_profile_for_variant(
    client: LLMClient,
    user_id: int,
    variant: dict[str, Any],
    *,
    max_publications: int = 30,
    max_interactions: int = 30,
    shared_evidence_cap: int = 30,
    min_interacted_articles: int = 1,
    fixed_priority_source_cap: int = 15,
    max_tokens: int = 220,
) -> dict[str, Any]:
    publications = get_user_publications(user_id, max_publications)
    summary = get_user_interactions(user_id, max_interactions)
    return generate_profile_text_from_evidence(
        client,
        user_id,
        variant,
        publications,
        summary,
        shared_evidence_cap=shared_evidence_cap,
        min_interacted_articles=min_interacted_articles,
        fixed_priority_source_cap=fixed_priority_source_cap,
        max_tokens=max_tokens,
    )
