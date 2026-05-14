# MSc Natural Language User Profile Generation

Python pipeline for generating natural-language user profiles from publication and interaction evidence and evaluating them in a temporal holdout setting.

## Scope

- fetch publication and interaction evidence from a compatible database
- generate five profile variants with different evidence-balancing strategies
- run temporal holdout evaluation with LLM-based judging
- export JSON reports for downstream analysis

## Repository Layout

- `profile_generation/db.py` - database configuration and connection setup
- `profile_generation/llm.py` - lightweight client for OpenAI-compatible and Ollama-style endpoints
- `profile_generation/generation.py` - evidence selection, prompt construction, and profile generation
- `profile_generation/temporal_experiment.py` - temporal split logic, holdout construction, judging, and ranking
- `scripts/generate_profiles_simple.py` - generate one or all profile variants for a user
- `scripts/run_temporal_experiment.py` - run the temporal evaluation pipeline and save JSON output

## Configuration

The scripts read database and model settings from environment variables. Start by copying `.env.example` to `.env` and filling in your own values.

Required database variables:

- `DB_USER`
- `DB_PASSWORD`
- `DB_HOST`
- `DB_NAME`

Recommended model variables:

- `LLM_GEN_BASE_URL`
- `LLM_JUDGE_API_BASE`
- `GEN_MODEL`
- `JUDGE_MODEL`
- `OPENAI_API_KEY` or `LLM_API_KEY`

This repository is not fully self-contained. End-to-end reproduction requires access to a database that follows the arXivDigest data model, including publication records and user interaction history.

## Usage

Generate all profile variants for one user:

```bash
python scripts/generate_profiles_simple.py --user-id 24 --run-all --output out/user24_profiles.json
```

Generate a single variant for one user:

```bash
python scripts/generate_profiles_simple.py --user-id 24 --variant-id weighted_x_0.80 --output out/user24_weighted_080.json
```

Run a temporal experiment for two users:

```bash
python scripts/run_temporal_experiment.py --users 24,30 --output out/temporal_report.json
```

