"""Generate short natural-language user profiles using an LLM.

Live generation mode is hybrid-only:
- hybrid (both publications and interactions)

Output is written to a JSON file or database, mostly json for testing 
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


DEFAULT_MODEL = "llama"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 220
MAX_PROMPT_PUBLICATIONS = 60
MISSING_ABSTRACT_PLACEHOLDER = "[ABSTRACT_MISSING_IN_DB]"
DEFAULT_SYSTEM_PROMPT = (
    "You produce concise first-person user-interest profiles for recommendation systems. "
    "Return plain text only."
)

ONE_SHOT_EXAMPLE = (
    "I am interested in a mix of long-term research themes and newer directions suggested by recent activity. "
    "I often work on concrete methods, applications, and evaluation questions that connect these interests. "
    "I care about clear problem definitions, practical impact, and how different topics fit together in my overall research profile."
)

@dataclass
class LLMConfig(LLMConnectionConfig):
    max_tokens: int = DEFAULT_MAX_TOKENS


class LLMClient:
    """Generation wrapper over the shared OpenAI-compatible LLM connector."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAICompatibleLLMClient(config)

    def generate(self, prompt: str, system_prompt: str) -> str:
        return self._client.chat_text(
            prompt,
            system_prompt,
            max_tokens=self.config.max_tokens,
        ).strip()


def _sanitize_profile_text(text: str, min_sentences: int = 3, max_sentences: int = 5) -> str:
    """Normalize model output to plain-text, first-person, 3-5 sentence profile."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+[.)]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Split into sentences and clamp to requested range.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    if len(sentences) < min_sentences and sentences:
        while len(sentences) < min_sentences:
            sentences.append(sentences[-1])

    if sentences:
        candidate = " ".join(sentences)
    else:
        candidate = "I am interested in research themes reflected in my publications and interactions."

    lower = candidate.lower()
    if not lower.startswith("i ") and not lower.startswith("i'm") and not lower.startswith("i am"):
        candidate = "I am interested in " + candidate[0].lower() + candidate[1:]

    return candidate.strip()


def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()])


def _normalize_profile_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _resolve_effective_weight(weight: float | None, default: float = 0.5) -> float:
    """Clamp weight to [0,1], with 0 treated as default balance."""
    if weight is None:
        return float(default)
    w = max(0.0, min(1.0, float(weight)))
    if w == 0.0:
        return float(default)
    return w


_TRAILING_CONNECTORS = {"and", "or", "for", "to", "in", "on", "of", "with", "from", "the", "a", "an"}


def _has_fragment_signals(text: str) -> bool:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    if not sentences:
        return True
    for s in sentences:
        tail = re.sub(r"[.!?]+$", "", s).strip().lower()
        if not tail:
            return True
        last_word_match = re.search(r"([a-zA-Z][a-zA-Z0-9\-]*)$", tail)
        if not last_word_match:
            return True
        if last_word_match.group(1) in _TRAILING_CONNECTORS:
            return True
    return False


def _finalize_profile_text(client: LLMClient, raw_text: str) -> str:
    """Ensure output is concise plain-text profile; rewrite once if needed."""
    candidate = _sanitize_profile_text(raw_text)
    lower = candidate.lower()
    bad_markers = [
        "main topics:",
        "trends:",
        "provided text",
        "here's",
        "1.",
        "2.",
        "3.",
        "it appears you've provided",
        "without a specific question",
        "i can offer some general insights",
        "summarizing the content",
        "identifying key papers",
    ]
    needs_rewrite = (
        _sentence_count(candidate) < 3
        or any(m in lower for m in bad_markers)
        or _has_fragment_signals(candidate)
    )
    if not needs_rewrite:
        return candidate

    rewrite_prompt = (
        "Revise the following text into a first-person user-interest profile in English. "
        "Return plain text only, exactly 4 sentences, with no headings, lists, or markdown."
        + "\n\n"
        f"Text to rewrite:\n{candidate}"
    )
    rewritten = client.generate(
        rewrite_prompt,
        "You revise user-interest profiles into concise first-person plain text.",
    )
    rewritten_clean = _sanitize_profile_text(rewritten)

    rewritten_lower = rewritten_clean.lower()
    rewritten_bad = any(m in rewritten_lower for m in bad_markers) or _has_fragment_signals(rewritten_clean)
    if rewritten_bad:
        return candidate

    return rewritten_clean


def _passes_quality_gate(profile_text: str) -> bool:
    text = (profile_text or "").strip()
    if not text:
        return False
    if _sentence_count(text) < 3:
        return False
    if _has_fragment_signals(text):
        return False

    low = text.lower()
    generic_markers = [
        "i am interested in research themes",
        "publications and interactions",
        "current research topics",
    ]
    if any(m in low for m in generic_markers):
        return False
    return True


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
        query = """
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

        cur.execute(query)
        rows = cur.fetchall()
        user_ids = [int(row[0]) for row in rows]

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
        rows = cur.fetchall()

    publications: list[dict[str, Any]] = []
    for row in rows:
        publications.append(
            {
                "title": row.get("title"),
                "abstract": row.get("abstract"),
                "year": row.get("year"),
                "venue": row.get("venue"),
                "citation_count": row.get("citation_count"),
                "fields_of_study": _safe_json_loads(row.get("fields_of_study")),
                "publication_types": _safe_json_loads(row.get("publication_types")),
            }
        )
    return publications


def _get_user_interaction_summary(user_id: int, max_examples: int) -> dict[str, Any]:
    with closing(get_connection()) as conn, closing(conn.cursor(dictionary=True)) as cur:
        cur.execute(
            """
            SELECT c.category_name, COUNT(*) AS cnt
            FROM article_feedback af
            JOIN article_categories ac ON ac.article_id = af.article_id
            JOIN categories c ON c.category_id = ac.category_id
            WHERE af.user_id = %s
              AND ((af.clicked_email IS NOT NULL) OR (af.clicked_web IS NOT NULL) OR (af.saved IS NOT NULL))
            GROUP BY c.category_name
            ORDER BY cnt DESC
            LIMIT 10
            """,
            (user_id,),
        )
        top_categories = cur.fetchall() or []

        cur.execute(
            """
            SELECT t.topic, COUNT(*) AS cnt
            FROM topic_recommendations tr
            JOIN topics t ON t.topic_id = tr.topic_id
            WHERE tr.user_id = %s
                            AND tr.clicked IS NOT NULL
            GROUP BY t.topic
            ORDER BY cnt DESC
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
                (af.saved IS NOT NULL) AS has_saved,
                ((af.clicked_email IS NOT NULL) OR (af.clicked_web IS NOT NULL)) AS has_clicked,
                GREATEST(
                  IFNULL(af.saved, '1000-01-01'),
                  IFNULL(af.clicked_email, '1000-01-01'),
                                    IFNULL(af.clicked_web, '1000-01-01')
                ) AS event_time,
                (
                  5 * (af.saved IS NOT NULL)
                                    + 3 * ((af.clicked_email IS NOT NULL) OR (af.clicked_web IS NOT NULL))
                ) AS engagement_score
            FROM article_feedback af
            JOIN articles a ON a.article_id = af.article_id
            WHERE af.user_id = %s
              AND (
                                af.saved IS NOT NULL OR af.clicked_email IS NOT NULL OR af.clicked_web IS NOT NULL
              )
            ORDER BY engagement_score DESC, event_time DESC
            LIMIT %s
            """,
            (user_id, max_examples),
        )
        top_engaged_articles = cur.fetchall() or []

    return {
        "top_categories": [
            {"category_name": row.get("category_name")}
            for row in top_categories
        ],
        "top_topics": [
            {"topic": row.get("topic")}
            for row in top_topics
        ],
        "recent_interactions": [
            {"title": row.get("title"), "event_time": str(row.get("event_time"))}
            for row in top_engaged_articles
        ],
        "top_engaged_articles": [
            {
                "title": row.get("title"),
                "abstract": row.get("abstract"),
                "has_saved": bool(row.get("has_saved") or False),
                "has_clicked": bool(row.get("has_clicked") or False),
                "event_time": str(row.get("event_time")),
            }
            for row in top_engaged_articles
        ],
    }


def _format_top_engaged_articles(summary: dict[str, Any], max_abstract_chars: int = 350) -> list[str]:
    lines: list[str] = []
    for idx, item in enumerate(summary.get("top_engaged_articles", []), start=1):
        title = item.get("title") or ""
        abstract = (item.get("abstract") or "").strip()
        if len(abstract) > max_abstract_chars:
            abstract = abstract[:max_abstract_chars] + "..."

        lines.append(
            f"{idx}. saved={item.get('has_saved', False)} | clicked={item.get('has_clicked', False)} | event_time={item.get('event_time')}"
        )
        lines.append(f"   title={title}")
        if abstract:
            lines.append(f"   abstract={abstract}")

    return lines


def _format_publication_entries(
    publications: list[dict[str, Any]],
    max_abstract_chars: int,
) -> list[str]:
    lines: list[str] = []
    for idx, pub in enumerate(publications, start=1):
        title = pub.get("title") or ""
        abstract = (pub.get("abstract") or "").strip()
        if len(abstract) > max_abstract_chars:
            abstract = abstract[:max_abstract_chars] + "..."
        if not abstract:
            abstract = MISSING_ABSTRACT_PLACEHOLDER
        year = pub.get("year")
        venue = pub.get("venue")
        citation_count = pub.get("citation_count")
        fos = pub.get("fields_of_study")

        lines.append(
            f"{idx}. title={title} | year={year} | venue={venue} | citations={citation_count}"
        )
        if fos:
            lines.append(f"   fields_of_study={fos}")
        lines.append(f"   abstract={abstract}")
    return lines


def _has_interaction_signal(summary: dict[str, Any]) -> bool:
    return (
        len(summary.get("top_engaged_articles", [])) > 0
        or len(summary.get("top_categories", [])) > 0
        or len(summary.get("top_topics", [])) > 0
    )


def _build_hybrid_prompt_publication_priority(
    user_id: int,
    publications_for_prompt: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"User ID: {user_id}",
        "Draft one concise first-person profile in English using both publication history and interaction behavior.",
        "Return plain text only, 3 to 5 sentences.",
        "Style example:",
        ONE_SHOT_EXAMPLE,
        "Do not copy phrases from the style example; infer specific topics from the evidence.",
        "Primary objective: the final profile should emphasize publication evidence.",
        "Sentence plan: sentences 1-3 should reflect publication trajectory; sentence 4 may mention interaction recency.",
        "Do not mention uncertainty, missing data, or AI status.",
        "Do not reference the prompt, data format, or model limitations.",
        "",
        "Evidence from publications (PRIMARY):",
    ]

    lines.extend(_format_publication_entries(publications_for_prompt, max_abstract_chars=400))

    lines.extend(
        [
            "",
            "Evidence from interactions (SECONDARY):",
            "Use article-level engagement evidence first (saved > clicked).",
            f"top_categories={summary.get('top_categories', [])}",
            f"top_topics={summary.get('top_topics', [])}",
            "Top engaged interaction articles:",
        ]
    )
    lines.extend(_format_top_engaged_articles(summary))
    return "\n".join(lines)


def _build_hybrid_prompt_interaction_priority(
    user_id: int,
    publications_for_prompt: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines: list[str] = [
        f"User ID: {user_id}",
        "Draft one concise first-person profile in English using both publication history and interaction behavior.",
        "Return plain text only, 3 to 5 sentences.",
        "Style example:",
        ONE_SHOT_EXAMPLE,
        "Do not copy phrases from the style example; infer specific topics from the evidence.",
        "Primary objective: the final profile should emphasize interaction evidence.",
        "Sentence plan: sentences 1-3 should reflect interaction behavior; sentence 4 may mention publication background.",
        "Do not mention uncertainty, missing data, or AI status.",
        "Do not reference the prompt, data format, or model limitations.",
        "",
        "Evidence from interactions (PRIMARY):",
        "Use article-level engagement evidence first (saved > clicked).",
        f"top_categories={summary.get('top_categories', [])}",
        f"top_topics={summary.get('top_topics', [])}",
        "Top engaged interaction articles:",
    ]
    lines.extend(_format_top_engaged_articles(summary))

    lines.extend(["", "Evidence from publications (SECONDARY):"])
    lines.extend(_format_publication_entries(publications_for_prompt, max_abstract_chars=400))
    return "\n".join(lines)


def _build_hybrid_prompt(
    user_id: int,
    publications: list[dict[str, Any]],
    summary: dict[str, Any],
    source_priority_instruction: str | None = None,
) -> str:
    publications_for_prompt = publications[:MAX_PROMPT_PUBLICATIONS]

    if source_priority_instruction:
        lower_rule = source_priority_instruction.lower()
        if "focus mainly on publication evidence" in lower_rule:
            return _build_hybrid_prompt_publication_priority(user_id, publications_for_prompt, summary)
        if "focus mainly on interaction evidence" in lower_rule:
            return _build_hybrid_prompt_interaction_priority(user_id, publications_for_prompt, summary)

    lines: list[str] = [
        f"User ID: {user_id}",
        "Draft one concise first-person profile in English using both publication history and interaction behavior.",
        "Return plain text only, 3 to 5 sentences.",
        "Style example:",
        ONE_SHOT_EXAMPLE,
        "Do not copy phrases from the style example; infer specific topics from the evidence.",
        "Do not mention uncertainty, missing data, or AI status.",
        "Do not reference the prompt, data format, or model limitations.",
        "Use only the evidence below.",
        "",
        "Evidence from interactions:",
        "Use article-level engagement evidence first (saved > clicked).",
        f"top_categories={summary.get('top_categories', [])}",
        f"top_topics={summary.get('top_topics', [])}",
        "Top engaged interaction articles:",
        "",
        "Evidence from publications:",
    ]

    if source_priority_instruction:
        lines.insert(
            8,
            f"When signals conflict, apply this priority rule: {source_priority_instruction}",
        )

    lines.extend(_format_top_engaged_articles(summary))

    lines.extend(_format_publication_entries(publications_for_prompt, max_abstract_chars=400))

    return "\n".join(lines)


def _generate_profile_from_hybrid_evidence(
    client: LLMClient,
    user_id: int,
    publications: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    publication_weight: float | None = None,
    source_priority_instruction: str | None = None,
    prompt_variant: str = "default",
    weighted_publication_cap: int | None = None,
    weighted_interaction_cap: int | None = None,
) -> tuple[str, dict[str, Any]]:
    has_interaction_signal = _has_interaction_signal(summary)
    has_publication_signal = len(publications) > 0

    if not has_publication_signal or not has_interaction_signal:
        raise ValueError("Hybrid mode requires both publication and interaction signals")

    prompt = _build_hybrid_prompt(
        user_id,
        publications,
        summary,
        source_priority_instruction=source_priority_instruction,
    )
    profile_text = _finalize_profile_text(client, client.generate(prompt, DEFAULT_SYSTEM_PROMPT))
    if not _passes_quality_gate(profile_text):
        quality_rewrite_prompt = (
            "Revise this user profile to improve clarity and specificity. "
            "Return plain text only, exactly 4 first-person sentences, with no bullets or headings.\n\n"
            f"Profile:\n{profile_text}"
        )
        improved = _finalize_profile_text(client, client.generate(quality_rewrite_prompt, DEFAULT_SYSTEM_PROMPT))
        if _passes_quality_gate(improved):
            profile_text = improved

    recent_publications = [
        {
            "title": pub.get("title"),
            "year": pub.get("year"),
            "venue": pub.get("venue"),
        }
        for pub in publications[:10]
    ]
    meta = {
        "mode": "hybrid",
        "prompt_variant": prompt_variant,
        "publication_weight": publication_weight,
        "weighted_publication_cap": weighted_publication_cap if weighted_publication_cap is not None else len(publications),
        "weighted_interaction_cap": weighted_interaction_cap if weighted_interaction_cap is not None else len(summary.get("top_engaged_articles", [])),
        "source_priority_instruction": source_priority_instruction,
        "publication_count": len(publications),
        "recent_publications": recent_publications,
        "has_interaction_signal": has_interaction_signal,
        "interaction_summary": summary,
    }
    return profile_text, meta


def _generate_profile_from_hybrid(
    client: LLMClient,
    user_id: int,
    max_publications: int,
    max_examples: int,
    publication_weight: float | None = None,
    source_priority_instruction: str | None = None,
    prompt_variant: str = "default",
) -> tuple[str, dict[str, Any]]:
    weighted_publication_cap = max_publications
    weighted_interaction_cap = max_examples

    if publication_weight is not None:
        publication_weight = _resolve_effective_weight(publication_weight)
        weighted_publication_cap = max(1, min(max_publications, int(round(max_publications * publication_weight))))
        weighted_interaction_cap = max(
            1,
            min(max_examples, int(round(max_examples * (1.0 - publication_weight)))),
        )

    publications = _get_user_publications(user_id, weighted_publication_cap)
    summary = _get_user_interaction_summary(user_id, weighted_interaction_cap)
    return _generate_profile_from_hybrid_evidence(
        client,
        user_id,
        publications,
        summary,
        publication_weight=publication_weight,
        source_priority_instruction=source_priority_instruction,
        prompt_variant=prompt_variant,
        weighted_publication_cap=weighted_publication_cap,
        weighted_interaction_cap=weighted_interaction_cap,
    )


def _resolve_output_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _persist_profiles_to_db(
    records: list[dict[str, Any]],
    prompt_id: str,
    llm: str,
    generated_at: datetime,
    run_id: str,
    include_failed: bool = False,
) -> int:
    """Persist generated profiles to DB table user_nl_profiles."""
    rows: list[tuple[Any, ...]] = []
    for record in records:
        status = record.get("status", "failed")
        if status != "success" and not include_failed:
            continue
        rows.append(
            (
                int(record["user_id"]),
                str(record.get("prompt_id") or prompt_id),
                str(record.get("llm") or llm),
                record.get("profile_text") if status == "success" else None,
                json.dumps(record.get("meta", {}), ensure_ascii=False) if status == "success" else None,
                status,
                record.get("error"),
                generated_at.replace(tzinfo=None),
                run_id,
            )
        )

    if not rows:
        return 0

    with closing(get_connection()) as conn, closing(conn.cursor()) as cur:
        cur.executemany(
            """
            INSERT INTO user_nl_profiles(
                user_id,
                prompt_id,
                llm,
                profile_text,
                meta_json,
                status,
                error_message,
                generated_at,
                run_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        conn.commit()

    return len(rows)


def _persist_experiment_candidates_to_db(
    records: list[dict[str, Any]],
    experiment_id: str,
    llm: str,
    generated_at: datetime,
    run_id: str,
) -> int:
    """Persist generated candidates into local experiment table."""
    rows: list[tuple[Any, ...]] = []
    for record in records:
        status = str(record.get("status", "failed"))
        variant_id = str(record.get("variant_id") or record.get("mode") or "unknown_variant")
        rows.append(
            (
                str(experiment_id),
                int(record["user_id"]),
                variant_id,
                str(record.get("prompt_id") or record.get("mode") or "hybrid"),
                str(record.get("llm") or llm),
                record.get("profile_text") if status == "success" else None,
                json.dumps(record.get("meta", {}), ensure_ascii=False) if status == "success" else None,
                status,
                record.get("error"),
                generated_at.replace(tzinfo=None),
                run_id,
            )
        )

    if not rows:
        return 0

    with closing(get_connection()) as conn, closing(conn.cursor()) as cur:
        cur.executemany(
            """
            INSERT INTO experiment_profile_candidates(
                experiment_id,
                user_id,
                variant_id,
                prompt_id,
                llm,
                profile_text,
                meta_json,
                status,
                error_message,
                generated_at,
                run_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                prompt_id = VALUES(prompt_id),
                llm = VALUES(llm),
                profile_text = VALUES(profile_text),
                meta_json = VALUES(meta_json),
                status = VALUES(status),
                error_message = VALUES(error_message),
                generated_at = VALUES(generated_at),
                run_id = VALUES(run_id)
            """,
            rows,
        )
        conn.commit()

    return len(rows)


def _hybrid_variant_specs(x: float) -> dict[str, dict[str, Any]]:
    """Return simplified hybrid variants: weighted + two prompt-priority variants."""
    x = _resolve_effective_weight(float(x))
    return {
        "weighted": {
            "variant_id": "weighted",
            "prompt_id": f"hybrid_weighted_{x:.2f}",
            "publication_weight": x,
            "source_priority_instruction": None,
            "prompt_variant": "weighted",
        },
        "prompt_publications_priority": {
            "variant_id": "prompt_publications_priority",
            "prompt_id": "hybrid_prompt_pub_priority",
            "publication_weight": None,
            "source_priority_instruction": (
                "the final profile should focus mainly on publication evidence; "
                "interaction evidence may only refine recency details"
            ),
            "prompt_variant": "prompt_publications_priority",
        },
        "prompt_interactions_priority": {
            "variant_id": "prompt_interactions_priority",
            "prompt_id": "hybrid_prompt_interactions_priority",
            "publication_weight": None,
            "source_priority_instruction": (
                "the final profile should focus mainly on interaction evidence; "
                "publication evidence may only provide background context"
            ),
            "prompt_variant": "prompt_interactions_priority",
        },
    }


def _resolve_normal_variant_spec(
    variant_id: str,
    publication_weight: float,
    variant_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve effective variant settings for normal generation."""
    if variant_id == "weighted":
        return {
            "prompt_id": "hybrid",
            "prompt_variant": "weighted",
            "publication_weight": _resolve_effective_weight(float(publication_weight)),
            "source_priority_instruction": None,
        }

    selected = variant_specs[variant_id]
    return {
        "prompt_id": str(selected["prompt_id"]),
        "prompt_variant": str(selected["prompt_variant"]),
        "publication_weight": selected["publication_weight"],
        "source_priority_instruction": selected["source_priority_instruction"],
    }


def main() -> None:
    load_project_env(PROJECT_ROOT)

    parser = argparse.ArgumentParser(
        description="Generate short NL user profiles and save as JSON."
    )
    parser.add_argument("--user-id", type=int, default=None, help="Generate profile for one user.")
    parser.add_argument("--limit-users", type=int, default=None, help="Limit number of users.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid"],
        default="hybrid",
        help="Evidence mode used for profile generation (hybrid only).",
    )
    parser.add_argument("--max-publications", type=int, default=30, help="Max publications per user.")
    parser.add_argument(
        "--max-interaction-examples",
        type=int,
        default=30,
        help="Max recent interacted article titles included as evidence.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scripts/data/user_nl_profiles_hybrid.json",
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--run-hybrid-4way",
        action="store_true",
        help=(
            "Generate all simplified hybrid variants for one user: weighted, publication-priority, interaction-priority."
        ),
    )
    parser.add_argument(
        "--hybrid-weight-x",
        type=float,
        default=0.8,
        help="Weight x in [0,1] used by weighted hybrid variant (0 => default 0.5).",
    )
    parser.add_argument(
        "--publication-weight",
        type=float,
        default=0.5,
        help=(
            "Primary weight for standard hybrid generation in [0,1]. "
            "If set to 0, defaults to 0.5. "
            "Controls evidence caps: publications=round(max_publications*x), "
            "interactions=round(max_interaction_examples*(1-x))."
        ),
    )
    parser.add_argument(
        "--variant-id",
        type=str,
        default="weighted",
        choices=[
            "weighted",
            "prompt_publications_priority",
            "prompt_interactions_priority",
        ],
        help=(
            "Variant used for normal generation. "
            "weighted uses --publication-weight (with 0 => 0.5 default); "
            "other values change prompt priority instruction only."
        ),
    )
    parser.add_argument(
        "--no-save-to-db",
        action="store_true",
        help="Disable persisting generated profiles to DB table user_nl_profiles.",
    )
    parser.add_argument(
        "--save-failed-to-db",
        action="store_true",
        help="Also persist failed generations to user_nl_profiles (default: only successful rows).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="",
        help="Optional experiment identifier. When set, records are also saved to experiment_profile_candidates.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_MODEL,
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default="",
        help="LLM API key override (preferred for non-local Ollama gateways).",
    )
    parser.add_argument(
        "--llm-api-url",
        type=str,
        default="",
        help="Full chat-completions URL override (e.g., https://host/v1/chat/completions).",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="",
        help="Base URL override (e.g., http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("LLM_TIMEOUT", "60")),
        help="LLM request timeout in seconds.",
    )

    args = parser.parse_args()

    if not (0.0 <= float(args.publication_weight) <= 1.0):
        raise ValueError("--publication-weight must be in [0,1].")

    model = (
        args.model
        if args.model != DEFAULT_MODEL
        else os.getenv("OLLAMA_MODEL")
        or os.getenv("LLM_MODEL")
        or DEFAULT_MODEL
    )
    api_url = (
        args.llm_api_url.strip()
        or os.getenv("OLLAMA_API_URL")
        or os.getenv("LLM_API_URL")
    )
    base_url = (
        args.llm_base_url.strip()
        or os.getenv("OLLAMA_HOST")
        or os.getenv("LLM_BASE_URL")
    )
    api_key = resolve_api_key(args.llm_api_key, endpoint_hint=f"{api_url} {base_url}")

    if not api_url and not base_url:
        raise ValueError(
            "LLM endpoint is required. Set OLLAMA_HOST/OLLAMA_API_URL or pass --llm-base-url/--llm-api-url."
        )

    client = LLMClient(
        LLMConfig(
            api_key=api_key,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_url=api_url,
            base_url=base_url,
            timeout=args.timeout,
        )
    )

    if args.user_id is not None:
        user_ids = [args.user_id]
    else:
        user_ids = _select_user_ids(args.limit_users)

    if args.run_hybrid_4way and args.user_id is None:
        raise ValueError("--run-hybrid-4way requires --user-id (single-user experiment).")

    if args.run_hybrid_4way and args.mode != "hybrid":
        raise ValueError("--run-hybrid-4way can only be used with --mode hybrid.")

    generated_at_dt = datetime.now(timezone.utc)
    generated_at = generated_at_dt.isoformat()
    run_id = str(uuid4())
    records: list[dict[str, Any]] = []
    variant_specs = _hybrid_variant_specs(float(args.hybrid_weight_x))

    for user_id in user_ids:
        if args.run_hybrid_4way:
            seen_profile_norm_to_variant: dict[str, str] = {}
            experiment_variants = list(variant_specs.values())

            for variant in experiment_variants:
                record = {
                    "user_id": user_id,
                    "status": "success",
                    "mode": "hybrid",
                    "variant_id": variant["variant_id"],
                    "prompt_id": variant["prompt_id"],
                    "llm": model,
                }

                try:
                    profile_text, meta = _generate_profile_from_hybrid(
                        client,
                        user_id,
                        args.max_publications,
                        args.max_interaction_examples,
                        publication_weight=variant["publication_weight"],
                        source_priority_instruction=variant["source_priority_instruction"],
                        prompt_variant=variant["prompt_variant"],
                    )

                    profile_norm = _normalize_profile_for_dedup(profile_text)
                    duplicate_of = seen_profile_norm_to_variant.get(profile_norm)
                    if duplicate_of:
                        raise ValueError(f"duplicate_profile_of:{duplicate_of}")
                    seen_profile_norm_to_variant[profile_norm] = variant["variant_id"]

                    record["profile_text"] = profile_text
                    record["meta"] = meta
                except Exception as exc:
                    record["status"] = "failed"
                    record["error"] = str(exc)

                records.append(record)
        else:
            selected = _resolve_normal_variant_spec(
                variant_id=args.variant_id,
                publication_weight=float(args.publication_weight),
                variant_specs=variant_specs,
            )

            record: dict[str, Any] = {
                "user_id": user_id,
                "status": "success",
                "mode": args.mode,
                "prompt_id": selected["prompt_id"],
                "llm": model,
            }

            try:
                profile_text, meta = _generate_profile_from_hybrid(
                    client,
                    user_id,
                    args.max_publications,
                    args.max_interaction_examples,
                    publication_weight=selected["publication_weight"],
                    source_priority_instruction=selected["source_priority_instruction"],
                    prompt_variant=selected["prompt_variant"],
                )
                record["profile_text"] = profile_text
                record["meta"] = meta

            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)

            records.append(record)

    output = {
        "generated_at": generated_at,
        "run_id": run_id,
        "mode": args.mode,
        "run_hybrid_4way": bool(args.run_hybrid_4way),
        "hybrid_weight_x": float(args.hybrid_weight_x),
        "publication_weight": float(args.publication_weight),
        "variant_id": args.variant_id,
        "model": args.model,
        "resolved_model": model,
        "user_count": len(user_ids),
        "records": records,
    }

    output_path = _resolve_output_path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    inserted_rows = 0
    if not args.no_save_to_db:
        inserted_rows = _persist_profiles_to_db(
            records=records,
            prompt_id=args.mode,
            llm=model,
            generated_at=generated_at_dt,
            run_id=run_id,
            include_failed=bool(args.save_failed_to_db),
        )

    experiment_rows = 0
    experiment_id = args.experiment_id.strip()
    if experiment_id:
        experiment_rows = _persist_experiment_candidates_to_db(
            records=records,
            experiment_id=experiment_id,
            llm=model,
            generated_at=generated_at_dt,
            run_id=run_id,
        )

    print(f"Saved profiles for {len(user_ids)} users to {output_path}")
    if not args.no_save_to_db:
        print(f"Persisted {inserted_rows} profile rows to DB table user_nl_profiles (run_id={run_id})")
    if experiment_id:
        print(
            "Persisted "
            f"{experiment_rows} candidate rows to DB table experiment_profile_candidates "
            f"(experiment_id={experiment_id}, run_id={run_id})"
        )


if __name__ == "__main__":
    main()
