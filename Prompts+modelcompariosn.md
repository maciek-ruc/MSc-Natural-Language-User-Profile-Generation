# Models used

- `gpt-4.1-mini`
- `gorina10.llama3.3:70b`

### System Prompt

```text
You write concise first-person research-interest profiles for recommendation systems. Return only the final profile as plain text. Start with 'I' or 'My'. Do not use headings, bullets, labels, or markdown.
```

### User Prompt 


```text
User ID: 2
Write one concise first-person research-interest profile in English.
Return plain text only, 3 to 4 sentences.
Use publication history and interaction behavior together to form one coherent profile.
Prefer specific themes, methods, domains, and evaluation concerns over broad umbrella labels.
Avoid generic phrases unless they are strongly supported by the evidence.
Interpret the JSON only as structured evidence about the user's research interests and activities.
Use only the content values in the JSON to infer themes, methods, tasks, domains, and evaluation concerns.
Do not describe the JSON structure, keys, arrays, field names, counts, nulls, or missing sections.
Do not write phrases such as 'based on the provided text', 'the JSON shows', 'this section is empty', 'there are N items', or 'the conversation appears to be'.
Do not list topics explicitly or explain the evidence; write the final profile itself.
Do not mention the prompt, the data, uncertainty, missing information, or formatting instructions.
Do not use headings, bullets, labels, or meta-commentary.
Start directly with 'I' or 'My'.
Style example:
I am interested in a coherent mix of long-term research themes and recent directions reflected in my activity. I focus on concrete methods, domains, and evaluation questions rather than generic topic labels. My profile should read like a compact description of sustained research interests.
Do not copy the example; infer the actual interests from the evidence below.

Evidence JSON:
{...actual evidence JSON shown in the dedicated evidence section below...}
```

## Returned Profiles Side by Side

These two profiles were generated from the same prompt shape and the same evidence payload.

| OpenAI (`gpt-4.1-mini`) | Ollama (`gorina10.llama3.3:70b`) |
|---|---|
| I focus on advancing user simulation techniques to improve the evaluation and training of conversational recommender and information access systems, emphasizing reproducibility and realistic user behavior modeling.<br><br>My work integrates agenda-based and large language model-driven simulators to address challenges in preference elicitation, conversational breakdowns, and system robustness.<br><br>I also develop benchmarks and platforms to standardize simulation-based evaluation, while exploring retrieval-augmented generation methods to enhance factual correctness and response grounding in conversational AI.<br><br>My research critically examines current evaluation practices and advocates for simulation as a key tool in progressing toward more intelligent, interactive AI systems. | I am interested in the provided JSON object contains a list of research articles related to information retrieval, conversational AI, and neural networks.<br><br>Here's a breakdown of the content: Articles: The object contains a list of 20 articles with their titles, abstracts, and other metadata.<br><br>Interaction Topics: There are 10 interaction topics listed, which seem to be related to the themes of the research articles.<br><br>These topics include neural information retrieval, conversational AI, entity retrieval, and semantic search. |

## Evidence Provided to the Generator

The profile generator sends a raw JSON evidence block directly inside the user prompt.

The evidence JSON has this structure:

```json
{
  "publications": [
    {
      "title": "...",
      "abstract": "...",
      "year": "...",
    }
  ],
  "interaction_topics": [
    {"topic": "..."}
  ],
  "interactions": [
    {
      "article_id": null,
      "title": "...",
      "abstract": "...",
      "event_time": null
    }
  ]
}
```

### What Each Evidence Section Means

- `publications`
  - The user's known publication history before holdout.
  - Each item includes title, truncated abstract, year.
- `interaction_topics`
  - Accepted user topics carried into the prompt as topic evidence.
  - In above example run, these are short topic labels such as `neural information retrieval`, `conversational ai`, and `semantic search`.
- `interactions`
  - Interaction-derived article evidence from the training side of the temporal split.
  - In this  `title` and `abstract` carry the semantic signal.

### Actual Evidence Size in This Run

For both runs, the evidence payload contained:

- `15` publications
- `10` interaction topics
- `15` interaction articles
- publication cap = `15`
- interaction cap = `15`
- weight `x = 0.50`

### Representative Evidence Examples

#### Example Publications

- `User Simulation in the Era of Generative AI: User Modeling, Synthetic Data Generation, and System Evaluation`
- `GINGER: Grounded Information Nugget-Based Generation of Responses`
- `Limitations of Current Evaluation Practices for Conversational Recommender Systems and the Potential of User Simulation`
- `UserSimCRS v2: Simulation-Based Evaluation for Conversational Recommender Systems`
- `SimLab: A Platform for Simulation-based Evaluation of Conversational Information Access Systems`

#### Example Interaction Topics

- `neural information retrieval`
- `information retrieval evaluation`
- `conversational ai`
- `entity retrieval`
- `semantic search`

#### Example Interaction Articles

- `$\texttt{Exoformer}$: Accelerating Bayesian atmospheric retrievals with transformer neural networks`
- `Negative Sampling Techniques in Information Retrieval: A Survey`
- `Enhancing Financial Report Question-Answering: A Retrieval-Augmented Generation System with Reranking Analysis`
- `PairSem: LLM-Guided Pairwise Semantic Matching for Scientific Document Retrieval`
- `Grounding Agent Memory in Contextual Intent`

## Prompts Used for LLM-as-Judge

The evaluation pipeline uses a separate LLM-as-judge stage in `scripts/run_temporal_split_experiment.py`. That stage does not generate profiles. Instead, it evaluates already generated profiles against held-out future evidence only.

### Judge System Prompt

The shared judge system prompt comes from `scripts/llm_connector.py`:

```text
You are an impartial and critical evaluator for user-profile quality. Return strict JSON only. No markdown.
```

### Pointwise Judge Prompt

The pointwise judge prompt is used to score one candidate profile at a time.

```text
Assess profile quality using held-out future evidence only.

User ID: {user_id}
Variant: {variant_id}
Profile:
{profile_text}

Held-out evidence JSON (interactions, accepted topics, and optionally publication holdout):
{holdout_json_pretty}

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
- relevance (1-5): overall alignment with the held-out evidence.
- specificity (1-5): concrete technical detail; non-generic wording.
- coverage (1-5): captures the breadth of important themes present in the held-out evidence.
- consistency (1-5): grammatical, semantic, and cross-sentence coherence.

Calibration guidance:
- Score 1-2 when criterion is clearly violated or mostly unsupported by evidence.
- Score 3 for partial match with noticeable gaps.
- Score 4 for strong match with minor issues.
- Score 5 only for clear, specific, and well-supported excellence.

Important evaluation policy:
- Use only the held-out evidence shown in this prompt.
- Do not assume access to any pre-holdout evidence beyond the profile text itself.
- Judge the profile only by how well it matches the held-out future evidence.

Mandatory failure rules:
1) If the text has malformed/clipped/ungrammatical sentences -> consistency=1 and overall=1 and pass_fail=fail.
2) If the text is semantically vague or awkward (example style: 'I am interested in knowledge, with a focus on resource.') -> specificity=1 and consistency=1 and overall=1 and pass_fail=fail.
3) If the text is generic boilerplate and could describe many unrelated users -> specificity<=2 and pass_fail=fail if severe.
Return strict JSON with keys: relevance,specificity,coverage,consistency,overall,pass_fail,rationale_short. Scores are 1.0-5.0. pass_fail is pass/fail.
```

### Pairwise Judge Prompt

The pairwise judge prompt is used when the pipeline compares two candidates directly.

```text
Compare two candidate user profiles using held-out future evidence only.

User ID: {user_id}
Candidate A ({variant_a}):
{profile_a}

Candidate B ({variant_b}):
{profile_b}

Held-out evidence JSON (interactions, accepted topics, and optionally publication holdout):
{holdout_json_pretty}

Use a pairwise rubric-based comparison protocol.
Evaluate A and B independently against the same rubric before making a final decision.
Choose the better profile using this priority: semantic quality/coherence first, then held-out evidence alignment, then specificity/coverage.
To reduce position bias, treat A/B labels as arbitrary and base judgment only on content quality and evidence alignment.
If one profile is awkward or semantically vague, it should lose.
Return strict JSON with keys: winner,confidence,reason_short where winner in {A,B,tie} and confidence in [0.0,1.0].
```

### Judge Evidence Inputs

The judge does not see only the final profile text. It also receives one explicit evidence payload:

- `heldout_evidence_json`: future evidence reserved for validation

In practice, that means the judge prompt embeds the held-out JSON block together with the scoring instructions.


