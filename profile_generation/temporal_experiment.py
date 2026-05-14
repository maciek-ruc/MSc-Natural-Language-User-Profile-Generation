from __future__ import annotations

from contextlib import closing
from datetime import datetime
import json
import random
from typing import Any

from profile_generation.db import get_connection
from profile_generation.generation import build_evidence_json, experiment_variants, generate_profile_text_from_evidence
from profile_generation.llm import LLMClient


DEFAULT_JUDGE_PUBLICATION_PAPERS = 2
DEFAULT_JUDGE_RANDOM_PAPERS = 2


def select_users(users_arg: str, max_users: int, holdout_n: int, min_train: int, required_publications: int) -> list[int]:
    if users_arg.strip():
        return [int(x.strip()) for x in users_arg.split(",") if x.strip()]
    with closing(get_connection()) as conn, closing(conn.cursor()) as cur:
        cur.execute(
            """
            SELECT af.user_id, COUNT(*) AS cnt,
                   (SELECT COUNT(*) FROM user_publications up WHERE up.user_id = af.user_id) AS pub_cnt
            FROM article_feedback af
            JOIN users u ON u.user_id = af.user_id AND NOT u.inactive
            WHERE af.saved IS NOT NULL OR af.clicked_email IS NOT NULL OR af.clicked_web IS NOT NULL
            GROUP BY af.user_id
            HAVING cnt >= %s AND pub_cnt >= %s
            ORDER BY af.user_id ASC
            """,
            (holdout_n + min_train, required_publications),
        )
        rows = [int(row[0]) for row in cur.fetchall()]
    return rows[: max_users] if max_users > 0 else rows


def fetch_interactions(user_id: int) -> list[dict[str, Any]]:
    with closing(get_connection()) as conn, closing(conn.cursor(dictionary=True)) as cur:
        cur.execute(
            """
            SELECT af.article_id, a.title, a.abstract,
                   GREATEST(IFNULL(af.saved, '1000-01-01'), IFNULL(af.clicked_email, '1000-01-01'), IFNULL(af.clicked_web, '1000-01-01')) AS event_time
            FROM article_feedback af
            JOIN articles a ON a.article_id = af.article_id
            WHERE af.user_id = %s
              AND (af.saved IS NOT NULL OR af.clicked_email IS NOT NULL OR af.clicked_web IS NOT NULL)
            ORDER BY event_time ASC
            """,
            (user_id,),
        )
        return cur.fetchall() or []


def build_split(interactions: list[dict[str, Any]], holdout_n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], datetime]:
    if len(interactions) <= holdout_n:
        raise ValueError("Not enough interactions for holdout")
    train = interactions[:-holdout_n]
    holdout = interactions[-holdout_n:]
    return train, holdout, holdout[0]["event_time"]


def fetch_publications_before(user_id: int, split_ts: datetime, max_publications: int) -> list[dict[str, Any]]:
    with closing(get_connection()) as conn, closing(conn.cursor(dictionary=True)) as cur:
        cur.execute(
            """
            SELECT p.title, p.abstract, p.year, p.venue, p.citation_count
            FROM user_publications up
            JOIN external_publications p ON p.publication_id = up.publication_id
            WHERE up.user_id = %s
              AND (up.created_at IS NULL OR up.created_at <= %s)
            ORDER BY COALESCE(p.year, 0) DESC, COALESCE(p.citation_count, 0) DESC, p.updated_at DESC
            LIMIT %s
            """,
            (user_id, split_ts, max_publications),
        )
        publications = cur.fetchall() or []
        if publications:
            return publications

        cur.execute(
            """
            SELECT p.title, p.abstract, p.year, p.venue, p.citation_count
            FROM user_publications up
            JOIN external_publications p ON p.publication_id = up.publication_id
            WHERE up.user_id = %s
            ORDER BY COALESCE(p.year, 0) DESC, COALESCE(p.citation_count, 0) DESC, p.updated_at DESC
            LIMIT %s
            """,
            (user_id, max_publications),
        )
        return cur.fetchall() or []


def interaction_summary_from_train(train: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    top = sorted(train, key=lambda row: row.get("event_time"), reverse=True)[:max_examples]
    return {
        "interacted_articles": [
            {
                "article_id": row.get("article_id"),
                "title": row.get("title"),
                "abstract": row.get("abstract"),
                "event_time": str(row.get("event_time")),
            }
            for row in top
        ]
    }


def build_holdout_items(holdout: list[dict[str, Any]], holdout_publications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interaction_items = [
        {
            "item_type": "interaction",
            "article_id": row.get("article_id"),
            "title": str(row.get("title") or "").strip(),
            "abstract": str(row.get("abstract") or "").strip(),
            "event_time": str(row.get("event_time")),
        }
        for row in holdout
        if str(row.get("title") or "").strip()
    ]
    publication_items = [
        {
            "item_type": "publication",
            "title": str(pub.get("title") or "").strip(),
            "abstract": str(pub.get("abstract") or "").strip(),
            "year": pub.get("year"),
            "venue": pub.get("venue"),
        }
        for pub in holdout_publications
        if str(pub.get("title") or "").strip()
    ]
    return interaction_items + publication_items


def fetch_random_article_papers(user_id: int, split_ts: datetime, excluded_article_ids: set[str], excluded_titles: set[str], limit: int) -> list[dict[str, Any]]:
    with closing(get_connection()) as conn, closing(conn.cursor(dictionary=True)) as cur:
        cur.execute(
            """
            SELECT a.article_id, a.title, a.abstract
            FROM articles a
            WHERE a.article_id IS NOT NULL
            ORDER BY a.article_id DESC
            LIMIT 500
            """
        )
        rows = cur.fetchall() or []
    candidates = []
    for row in rows:
        article_id = str(row.get("article_id") or "").strip().lower()
        title = str(row.get("title") or "").strip().lower()
        if not title or article_id in excluded_article_ids or title in excluded_titles:
            continue
        candidates.append(row)
    random.Random(f"{user_id}|{split_ts}").shuffle(candidates)
    return candidates[:limit]


def build_judge_papers(
    user_id: int,
    split_ts: datetime,
    interactions: list[dict[str, Any]],
    known_publications: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    judge_publication_papers: int = DEFAULT_JUDGE_PUBLICATION_PAPERS,
    judge_random_papers: int = DEFAULT_JUDGE_RANDOM_PAPERS,
) -> list[dict[str, Any]]:
    publication_items = [pub for pub in known_publications if str(pub.get("title") or "").strip()][:judge_publication_papers]
    holdout_items = [row for row in holdout if str(row.get("title") or "").strip()][:2]
    excluded_article_ids = {str(row.get("article_id") or "").strip().lower() for row in interactions if row.get("article_id")}
    excluded_titles = {str(item.get("title") or "").strip().lower() for item in publication_items + holdout_items if item.get("title")}
    random_items = fetch_random_article_papers(user_id, split_ts, excluded_article_ids, excluded_titles, judge_random_papers)

    papers: list[dict[str, Any]] = []
    for pub in publication_items:
        papers.append(
            {
                "paper_id": f"paper_{len(papers) + 1}",
                "paper_role": "publication",
                "title": pub.get("title"),
                "abstract": pub.get("abstract"),
                "year": pub.get("year"),
                "venue": pub.get("venue"),
            }
        )
    for row in holdout_items:
        papers.append(
            {
                "paper_id": f"paper_{len(papers) + 1}",
                "paper_role": "holdout_interaction",
                "article_id": row.get("article_id"),
                "title": row.get("title"),
                "abstract": row.get("abstract"),
                "event_time": str(row.get("event_time")),
            }
        )
    for row in random_items:
        papers.append(
            {
                "paper_id": f"paper_{len(papers) + 1}",
                "paper_role": "random",
                "article_id": row.get("article_id"),
                "title": row.get("title"),
                "abstract": row.get("abstract"),
            }
        )
    return papers


def judge_prompt(user_id: int, variant_id: str, profile_text: str, judge_papers: list[dict[str, Any]]) -> str:
    papers_json = json.dumps(judge_papers, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            f"User ID: {user_id}",
            f"Variant: {variant_id}",
            "Evaluate whether the profile matches the paper set.",
            "Return strict JSON with keys relevance, specificity, coverage, consistency, publication_mean, holdout_mean, random_mean, overall, rationale_short.",
            "Use 1-5 scores.",
            "Profile:",
            profile_text,
            "",
            "Papers JSON:",
            papers_json,
        ]
    )


def parse_judge_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Judge response is not JSON")
    return json.loads(stripped[start : end + 1])


def derive_profile_judge_scores(judge_raw: dict[str, Any]) -> dict[str, Any]:
    publication_mean = float(judge_raw.get("publication_mean", judge_raw.get("overall", 0)))
    holdout_mean = float(judge_raw.get("holdout_mean", judge_raw.get("overall", 0)))
    random_mean = float(judge_raw.get("random_mean", 0))
    overall = float(judge_raw.get("overall", max(publication_mean, holdout_mean)))
    return {
        "relevance": float(judge_raw.get("relevance", 0)),
        "specificity": float(judge_raw.get("specificity", 0)),
        "coverage": float(judge_raw.get("coverage", 0)),
        "consistency": float(judge_raw.get("consistency", 0)),
        "overall": overall,
        "rationale_short": str(judge_raw.get("rationale_short", "")).strip(),
        "publication_mean": publication_mean,
        "holdout_mean": holdout_mean,
        "random_mean": random_mean,
        "pass_fail": "pass" if publication_mean > random_mean and holdout_mean > random_mean else "fail",
    }


def ranking_score(item: dict[str, Any]) -> float:
    judge = item.get("judge") or {}
    return float(judge.get("publication_mean", 0.0)) + float(judge.get("holdout_mean", 0.0)) - 2.0 * float(judge.get("random_mean", 0.0))


def run_experiment_for_user(
    *,
    user_id: int,
    gen_client: LLMClient,
    judge_client: LLMClient,
    holdout_n: int,
    holdout_publications: int,
    max_publications: int,
    max_interaction_examples: int,
    shared_evidence_cap: int,
    min_interacted_articles: int,
    fixed_priority_source_cap: int,
    top_k: int,
) -> dict[str, Any]:
    interactions = fetch_interactions(user_id)
    train, holdout, split_ts = build_split(interactions, holdout_n)
    known_publications = fetch_publications_before(user_id, split_ts, max_publications + holdout_publications)
    holdout_publication_items = known_publications[:holdout_publications]
    train_publications = known_publications[holdout_publications : holdout_publications + max_publications]
    train_summary = interaction_summary_from_train(train, max_interaction_examples)
    judge_papers = build_judge_papers(user_id, split_ts, interactions, train_publications, holdout)
    holdout_items = build_holdout_items(holdout, holdout_publication_items)

    generated_variants = []
    for variant in experiment_variants():
        try:
            record = generate_profile_text_from_evidence(
                gen_client,
                user_id,
                variant,
                train_publications,
                train_summary,
                shared_evidence_cap=shared_evidence_cap,
                min_interacted_articles=min_interacted_articles,
                fixed_priority_source_cap=fixed_priority_source_cap,
            )
        except Exception as exc:
            record = {
                "user_id": user_id,
                "variant_id": variant["variant_id"],
                "prompt_id": variant["prompt_id"],
                "status": "failed",
                "error": str(exc),
            }
        generated_variants.append(record)

    scored = []
    for record in generated_variants:
        if record.get("status") != "success":
            continue
        raw = parse_judge_json(
            judge_client.chat_text(
                judge_prompt(user_id, str(record["variant_id"]), str(record["profile_text"]), judge_papers),
                max_tokens=700,
            )
        )
        record["judge"] = derive_profile_judge_scores(raw)
        scored.append(record)

    passed = [record for record in scored if str(record.get("judge", {}).get("pass_fail")) == "pass"]
    ranked_pool = passed if passed else scored
    ranked = sorted(ranked_pool, key=ranking_score, reverse=True)

    return {
        "user_id": user_id,
        "status": "success" if ranked else "failed",
        "split_ts": str(split_ts),
        "known_evidence_json": build_evidence_json(train_publications, train_summary),
        "holdout_items": holdout_items,
        "judge_papers": judge_papers,
        "generated_variants": generated_variants,
        "top_selected": [
            {
                "variant_id": record["variant_id"],
                "prompt_id": record["prompt_id"],
                "overall": record.get("judge", {}).get("overall"),
            }
            for record in ranked[: max(top_k, 1)]
        ],
    }
