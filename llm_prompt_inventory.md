# LLM Prompts Finalized


## Prompt table

| # | Prompt name | Brief overview | Motivation | Full prompt |
|---|---|---|---|---|
| 1 | Generation system prompt (`DEFAULT_SYSTEM_PROMPT`) | Sets style and output constraints globally for profile generation. | Reduces format drift and keeps outputs plain-text and consistent. | See Prompt 1 below |
| 2 | Shared style example (`ONE_SHOT_EXAMPLE`) | One-shot style anchor (not content anchor). | Helps keep the same style. It used to be a little more literal example but weaker models just copied it despite stating it's forbidden. | See Prompt 2 below |
| 3 | Hybrid prompt (default weighted evidence) | Combines long-term publication signal with recent interaction signal. | By default it takes 30 records from publications and 30 from interaction data, x variable helps sway the proportions (ex. x=0.8 publications included=24 interactions included=6), We use default variant with no weights, one with x=0.8 and one with x=0.2 | See Prompt 3 below |
| 4 | Hybrid prompt: publications-priority variant | Publication-led framing with interactions as secondary. | Tests if publication being underlined via prompt helps the profile outcome | See Prompt 4 below |
| 5 | Hybrid prompt: interactions-priority variant | Interaction-led framing with publications as secondary. | Tests recency-heavy personalization. | See Prompt 5 below |
| 6 | Sanity rewrite prompt | Repair pass for malformed/generic output. | Fixes weak generations without rebuilding full evidence prompt. | See Prompt 6 below |
| 7 | Quality rewrite prompt | Specificity/clarity improvement pass. | Raises quality before candidate acceptance/ranking. | See Prompt 7 below |
| 8 | Judge system prompt | Forces strict machine-readable judging output. | Prevents markdown/prose leakage in scoring pipeline. | See Prompt 8 below |
| 9 | Pointwise judge user prompt | Rubric-based independent scoring of one candidate. | Converts quality to comparable numeric signals. | See Prompt 9 below |
| 10 | Pairwise judge user prompt | Head-to-head comparison between two candidates. | Stabilizes ranking when pointwise scores are close. | See Prompt 10 below |

## Full prompts

### Prompt 1 — Generation system prompt

```text
You produce concise first-person user-interest profiles for recommendation systems. Return plain text only.
```

### Prompt 2 — Shared style example

```text
I am interested in a mix of long-term research themes and newer directions suggested by recent activity. I often work on concrete methods, applications, and evaluation questions that connect these interests. I care about clear problem definitions, practical impact, and how different topics fit together in my overall research profile.
```

### Prompt 3 — Hybrid prompt (default weighted evidence)

```text
Draft one concise first-person profile in English using both publication history and interaction behavior.
Return plain text only, 3 to 5 sentences.
Style example:
{ONE_SHOT_EXAMPLE}
Do not copy phrases from the style example; infer specific topics from the evidence.
Do not mention uncertainty, missing data, or AI status.
Do not reference the prompt, data format, or model limitations.
Use only the evidence below.
{OPTIONAL: When signals conflict, apply this priority rule: {source_priority_instruction}}

Evidence from interactions:
Use article-level engagement evidence first (saved > clicked).
top_categories={summary.top_categories}
top_topics={summary.top_topics}
Top engaged interaction articles:
{for each item i:}
{i}. saved={has_saved} | clicked={has_clicked} | event_time={event_time}
	title={title}
	abstract={abstract<=350 chars if present}

Evidence from publications:
{for each publication i:}
{i}. title={title} | year={year} | venue={venue} | citations={citation_count}
	fields_of_study={fields_of_study if present}
	abstract={abstract<=400 chars OR [ABSTRACT_MISSING_IN_DB]}
```

### Prompt 4 — Hybrid prompt: publications-priority variant

```text
User ID: {user_id}
Draft one concise first-person profile in English using both publication history and interaction behavior.
Return plain text only, 3 to 5 sentences.
Style example:
{ONE_SHOT_EXAMPLE}
Do not copy phrases from the style example; infer specific topics from the evidence.
Primary objective: the final profile should emphasize publication evidence.
Sentence plan: sentences 1-3 should reflect publication trajectory; sentence 4 may mention interaction recency.
Do not mention uncertainty, missing data, or AI status.
Do not reference the prompt, data format, or model limitations.

Evidence from publications (PRIMARY):
{publication entries, abstract<=400}

Evidence from interactions (SECONDARY):
Use article-level engagement evidence first (saved > clicked).
top_categories={...}
top_topics={...}
Top engaged interaction articles:
{engaged article entries}
```

### Prompt 5 — Hybrid prompt: interactions-priority variant

```text
User ID: {user_id}
Draft one concise first-person profile in English using both publication history and interaction behavior.
Return plain text only, 3 to 5 sentences.
Style example:
{ONE_SHOT_EXAMPLE}
Do not copy phrases from the style example; infer specific topics from the evidence.
Primary objective: the final profile should emphasize interaction evidence.
Sentence plan: sentences 1-3 should reflect interaction behavior; sentence 4 may mention publication background.
Do not mention uncertainty, missing data, or AI status.
Do not reference the prompt, data format, or model limitations.

Evidence from interactions (PRIMARY):
Use article-level engagement evidence first (saved > clicked).
top_categories={...}
top_topics={...}
Top engaged interaction articles:
{engaged article entries}

Evidence from publications (SECONDARY):
{publication entries, abstract<=400}
```

### Prompt 6 — Sanity rewrite prompt

```text
System: You revise user-interest profiles into concise first-person plain text.
User: Revise the following text into a first-person user-interest profile in English. Return plain text only, exactly 4 sentences, with no headings, lists, or markdown.

Text to rewrite:
{candidate}
```

### Prompt 7 — Quality rewrite prompt

```text
System: You produce concise first-person user-interest profiles for recommendation systems. Return plain text only.
User: Revise this user profile to improve clarity and specificity. Return plain text only, exactly 4 first-person sentences, with no bullets or headings.

Profile:
{profile_text}
```

### Prompt 8 — Judge system prompt

```text
You are an impartial critical evaluator. Return strict JSON only.
```

### Prompt 9 — Pointwise judge user prompt

```text
Assess profile quality against held-out future interactions.

User ID: {user_id}
Variant: {variant_id}
Profile:
{profile_text}

Held-out interactions (title + abstract):
{for each holdout i:}
{i}. title={title}
	abstract={abstract<=500 chars if present}

Scoring rubric:
Evaluate the profile using four dimensions: relevance, specificity, coverage, and consistency.
For each dimension and overall, assign an INTEGER score from 1 to 5 only.
Level meanings (must use exactly this scale):
1 = very poor / fails criterion
2 = weak / major issues
3 = acceptable / mixed quality
4 = strong / minor issues
5 = excellent / fully satisfies criterion

Dimensions:
- relevance (1-5): alignment with held-out topics/tasks.
- specificity (1-5): concrete technical detail; non-generic wording.
- coverage (1-5): captures breadth of important held-out interests.
- consistency (1-5): grammatical, semantic, and cross-sentence coherence.

Calibration guidance:
- Score 1-2 when criterion is clearly violated or mostly unsupported by evidence.
- Score 3 for partial match with noticeable gaps.
- Score 4 for strong match with minor issues.
- Score 5 only for clear, specific, and well-supported excellence.

Mandatory failure rules:
1) If the text has malformed/clipped/ungrammatical sentences -> consistency=1 and overall=1 and pass_fail=fail.
2) If the text is semantically vague or awkward (example style: 'I am interested in knowledge, with a focus on resource.') -> specificity=1 and consistency=1 and overall=1 and pass_fail=fail.
3) If the text is generic boilerplate and could describe many unrelated users -> specificity<=2 and pass_fail=fail if severe.
Return strict JSON with keys: relevance,specificity,coverage,consistency,overall,pass_fail,rationale_short. Scores are 1.0-5.0. pass_fail is pass/fail.
```

### Prompt 10 — Pairwise judge user prompt

```text
Compare two candidate user profiles against held-out future interactions.

User ID: {user_id}
Candidate A ({variant_a}):
{profile_a}

Candidate B ({variant_b}):
{profile_b}

Held-out interactions (title + abstract):
{holdout items}

Evaluate A and B independently against the same rubric before making a final decision.
Choose the better profile using this priority: semantic quality/coherence first, then relevance, then specificity/coverage.
To reduce position bias, treat A/B labels as arbitrary and base judgment only on content quality and evidence alignment.
If one profile is awkward or semantically vague, it should lose.
Return strict JSON with keys: winner,confidence,reason_short where winner in {A,B,tie} and confidence in [0.0,1.0].
```

## Request envelope actually sent to the API

Each LLM call is sent as a 2-message chat payload:
- `system`: one of the system prompts above
- `user`: one of the user prompts above

So full on-wire prompt is always:
- `messages[0].content = system_prompt`
- `messages[1].content = prompt`

