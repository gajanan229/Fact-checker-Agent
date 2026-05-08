# Veritas AI — Code Review & Recommendations

This document is a focused, opinionated review of the agent and UI as they stand today. It calls out concrete bugs, addresses each of your stated concerns, and recommends a set of refactors and deletions. No code is written here — this is a plan.

---

## TL;DR

The pipeline runs end-to-end, but several stages are doing far less than they appear to. The biggest issues:

1. **Evidence research is not "research"** — it's one fixed Tavily query per claim. No reasoning, no follow-up, no adjudication. Claims never actually get marked verified/debunked.
2. **Critique has no access to the world** — it's a pure LLM self-grade against the response and its own claims, with no web verification.
3. **The frontend is reading from fields the backend never sets**, so the Evidence Locker's analysis text and per-claim sources are silently empty for every run.
4. **`TargetDisplay` and parts of `RedTeamLog` are placeholders** — they hardcode copy or only show one field of a richer critique object.
5. **~3,500+ lines of dead code** sit in `core/hitl.py`, `core/security.py`, `nodes/human_interaction.py`, and large parts of `core/graph.py`, `core/serialization.py`, `core/visualization.py`. The autonomous graph bypasses all of it.

Everything below is detail and prescription.

---

## 1. Agent concerns (your list, addressed)

### 1.1 "Claims are limited to 3 — too shallow"

**Diagnosis:** There is no hard `3` cap in the production code path. What you're observing is a side-effect of the prompt design, not a literal limit.

- [`claim_identification.py`](Backend-Agent/src/veritas_ai/nodes/claim_identification.py) tells the LLM `"Be conservative — only extract claims that are clearly factual and verifiable"` and forces a rigid `claim_type` enum (`factual | statistical | historical | scientific | other`). Combined with `confidence < 0.3` filtering and the dedup heuristic in `ClaimProcessor.filter_duplicate_claims`, typical short TikTok transcripts produce 2–4 surviving claims.
- The literal `3` in the system is `EvidenceResearcher.max_sources_per_claim = 3` (sources per claim, not number of claims) and `create_test_claims(count=3)` (a dev fixture).

**Recommendation:**
- **Drop the "be conservative" framing.** Replace with: *extract every distinct, externally verifiable claim. Prefer recall over precision; later stages will filter.*
- **Drop the `claim_type` enum**, or reduce to a free-text string. The downstream pipeline never branches on it; it's overhead the LLM has to satisfy.
- **Loosen the confidence floor** from `0.3` to something lower (or drop entirely) and let the research stage be the actual filter.
- **Tighten dedup**, not the extractor. The current `ClaimProcessor.filter_duplicate_claims` is fine for catching exact duplicates but the 60% word-overlap threshold can collapse legitimately distinct claims.
- Keep an upper bound (e.g. 15–20 claims) only as a safety rail, not as the primary control.

### 1.2 "EvidenceResearcher just loops — give the LLM the tool"

**Diagnosis:** Confirmed and central. [`_research_single_claim`](Backend-Agent/src/veritas_ai/nodes/evidence_research.py#L75) issues exactly one search with the literal query `f"{claim['text']} fact check"`, dumps Tavily's `answer` into `evidence_summary`, marks the claim `RESEARCHING`, and exits. There's no reasoning about query strategy, no follow-up search, and **the claim never reaches a verdict** (`VERIFIED`, `DEBUNKED`, `MISLEADING`, `LACKS_CONTEXT`, `UNVERIFIABLE`). Status stays at `RESEARCHING` permanently.

This is also the upstream cause of UI bug #3 (Evidence Locker verdict colors fall through to "default gray" because `RESEARCHING` isn't in `getVerdictColor`).

**Recommendation — replace this whole node with a tool-using sub-agent.** Two viable shapes:

- **Option A (simpler): LangGraph ReAct agent per claim.** Use `langgraph.prebuilt.create_react_agent` with tools: `tavily_search`, `tavily_extract` (full-page fetch), and a final `submit_verdict` tool that writes structured output. The agent reasons, may issue several queries, may rephrase, may follow up on a specific source, then commits a verdict + summary + chosen sources.
- **Option B (more controllable): tiny sub-graph.** Nodes: `plan_queries → search → assess_sufficiency → (loop or) adjudicate`. Bound iterations (e.g. max 3 search rounds per claim) for cost.

Either way, the contract this node must satisfy:
- Set `claim['status']` to one of the **terminal** statuses (`VERIFIED`, `DEBUNKED`, `MISLEADING`, `LACKS_CONTEXT`, `UNVERIFIABLE`). Stop using `RESEARCHING` as a final state.
- Write a real `verification_summary` (the agent's reasoned conclusion) **and** `evidence_summary` (the supporting evidence). Currently only the latter exists, copied from Tavily's auto-answer.
- Cap parallelism (the current `asyncio.gather` over all claims will blow through Tavily quotas on a long claim list — add a semaphore).

### 1.3 "critique_response should be a sub-graph with web search access"

**Diagnosis:** Agreed. Today [`adversarial_review.py`](Backend-Agent/src/veritas_ai/nodes/adversarial_review.py) is two LLM calls (quality assessment + bias detection) over the draft response and its own claim list. It has no way to detect a hallucinated statistic the response just introduced. It is a self-grade, not a red team.

The 8 quality dimensions × 9 bias categories × revision-recommendation generator is also overbuilt for the value it produces. Most of the dimension-level scoring never makes it to the UI — `RedTeamLog` only displays `feedback_text`.

**Recommendation — convert to a small sub-graph:**

```
critique_subgraph:
  extract_response_claims  (LLM: pull every fact-y assertion out of the draft)
        │
        ▼
  verify_response_claims   (tool-using agent: web-search each one, like 1.2)
        │
        ▼
  judge                    (LLM: combine verification results + style/tone/bias
                            into a single Critique with revision_needed flag)
```

Important details:
- The new red-team is checking the **response itself**, not just re-grading the upstream claims. That's where the value is — catching things the response generator added or distorted.
- Collapse the 8 quality dimensions into ~3 that you actually surface (factual accuracy, tone/respectfulness, citation quality). Drop separate bias detection as its own LLM call — fold it into `judge` as one of those dimensions. You'll save a Gemini call per critique pass.
- Keep the revision loop, but cap it firmly at `max_revisions=2`. The current cap is 3 with a fallback, and in practice you almost never recover quality on a 3rd pass; you just burn tokens.
- Pass critique state through to the UI in full (see UI section).

### 1.4 "Prompts should be cleaned up"

Across the four LLM-using nodes, the prompts have drifted into a few common antipatterns:

- **Over-prescriptive constraints that the structured output already enforces.** [`response_generation.py:268-275`](Backend-Agent/src/veritas_ai/nodes/response_generation.py#L268-L275) hard-codes `Must be AT LEAST 100 words long` while the Pydantic validator in the same file caps the response at 2000 chars — these tug against each other. The Pydantic schema is enough; drop the natural-language word-count.
- **Duplicate format instructions.** `PydanticOutputParser.get_format_instructions()` is injected as `{format_instructions}`, then the prompt also enumerates the fields in prose. Keep one or the other; prefer letting Gemini's native tool/structured output do the work and remove `PydanticOutputParser` entirely (Gemini 2.x and 3.x support `with_structured_output`).
- **System prompts that lecture the model.** [`adversarial_review.py:170-192`](Backend-Agent/src/veritas_ai/nodes/adversarial_review.py#L170-L192) is two screens of "be especially vigilant for…" That tone tends to push Gemini toward over-flagging (every response gets a critique, every critique recommends revisions). Tighten to a one-paragraph role + a short rubric.
- **Claim extractor's `opinion_words` filter** ([`claim_identification.py:60-64`](Backend-Agent/src/veritas_ai/nodes/claim_identification.py#L60-L64)) is a brittle blocklist. A claim like *"experts believe X"* is a legitimate verifiable claim about what experts say; this filter caps its confidence and marks it unverifiable. Remove this and trust the LLM's own assessment.
- **Transcript cleaner's prompt** is fine, but the whole node may be unnecessary — the Apify TikTok scraper actor returns clean text most of the time, and Gemini's later stages can handle minor noise. A measurement worth taking: how often does the cleaner actually change the transcript meaningfully? If rarely, delete the node and save a Gemini call per run.

**General principle:** every prompt should answer three questions in order — *what role*, *what input you'll get*, *what output shape*. Anything else is noise. Aim for system prompts under 10 lines.

---

## 2. UI concerns (your list, addressed)

### 2.1 `TargetDisplay` is a placeholder

**Diagnosis:** Confirmed. [`TargetDisplay.tsx:26-27`](Frontend/src/components/workbench/TargetDisplay.tsx#L26-L27) literally hardcodes:

```
<h4>TikTok Video Analysis</h4>
<p>Content: Health claims about celery juice</p>
```

The component receives only `videoUrl` as a prop. None of the actual extracted metadata (title, transcript snippet, claims-found count) is plumbed through.

**Recommendation:**
- Add `video_metadata`, a `transcript_excerpt` (first ~200 chars of cleaned transcript), and `claims_count` to the `caseFile` payload returned by `transform_state_for_frontend`.
- Update `TargetDisplay` props to take `videoUrl`, `metadata`, `excerpt`, `claimsCount` and render them. Drop the celery-juice copy.
- Make the URL clickable (`<a href={videoUrl} target="_blank">`).
- The 24×24 `Play` icon "thumbnail" can stay as a stylized placeholder — TikTok thumbnails require their oEmbed endpoint which is fine to skip for v1.

### 2.2 Red Team Log doesn't really work

**Diagnosis:** Two real problems:

1. [`RedTeamLog.tsx:74-80`](Frontend/src/components/workbench/RedTeamLog.tsx#L74-L80) shows `"Quality checks passed"` **unconditionally** when collapsed, regardless of whether `is_revision_needed` was true.
2. The backend's `Critique` object has rich structure (quality dimension scores, detected biases, prioritized recommendations, strengths, critical issues, overall score). The frontend only renders `feedback_text`, which is itself a pre-formatted string built server-side. So the structured data is computed and discarded.

**Recommendation:**
- Stop hand-formatting `feedback_text` server-side. Pass the critique's structured fields through the API: `overall_quality_score`, `quality_scores[]`, `detected_biases[]`, `revision_recommendations[]`, `strengths[]`, `critical_issues[]`, `is_revision_needed`, `revision_count`.
- Re-render `RedTeamLog` with three sections: a status banner (passed / revised N times / blocked), a dimension-score bar chart (3 bars — accuracy, tone, citations), and a collapsible details list (recommendations + biases). Drop `feedback_text` entirely.
- The "passed" badge should reflect `is_revision_needed === false`, not be a static decoration.
- After the critique sub-graph rewrite (1.3), this also gets a "Verified claims in response" row showing which response-statements were web-checked and what was found — this is the actual red-team output and is what makes this panel valuable.

### 2.3 Evidence Locker analysis section is empty

**Diagnosis:** **This is a backend bug, not a frontend bug.** [`api/utils.py`](Backend-Agent/src/veritas_ai/api/utils.py) is reading from fields that don't exist on the state object:

| Code | Issue |
|---|---|
| `graph_state.get("sources", [])` ([line 14](Backend-Agent/src/veritas_ai/api/utils.py#L14)) | Sources are nested inside each claim (`claim["sources"]`), not at the top level of state. Always returns `[]`. |
| `source_map = {source["id"]: ...}` ([line 17](Backend-Agent/src/veritas_ai/api/utils.py#L17)) | `Source` TypedDict has no `id` field (see [state.py:41-52](Backend-Agent/src/veritas_ai/core/state.py#L41-L52)). Map is empty. |
| `claim.get("source_ids", [])` ([line 34](Backend-Agent/src/veritas_ai/api/utils.py#L34)) | `Claim` has `sources`, not `source_ids`. Always returns `[]`. |
| `claim.get("summary", "No summary available.")` ([line 41](Backend-Agent/src/veritas_ai/api/utils.py#L41)) | `Claim` has `evidence_summary` and `verification_summary`, not `summary`. Always returns the fallback. |

Net effect: **for every run**, every dossier entry gets `summary: "No summary available."` and `sources: []`. That's exactly your observed UI behavior.

There's also a secondary bug in [`EvidenceLocker.tsx`](Frontend/src/components/workbench/EvidenceLocker.tsx): the `Claim` interface declares `claim_id: number` but `WorkbenchPage.tsx` correctly uses `string` (matching the backend's `claim_id_str`). The component's `expandedClaims: number[]` and `toggleClaim(claimId: number)` will technically work because JS coerces — but it's a type bug waiting to bite.

**Recommendation:**
- Rewrite `transform_state_for_frontend` to read the actual state shape:
  - `summary` ← `claim['verification_summary']` (the agent's adjudication) **plus** `claim['evidence_summary']` (raw evidence). Concatenate or send both as separate fields.
  - `sources` ← `claim['sources']` directly. Map each `Source` to `{ url, title, domain, content_snippet }` for the frontend (no `id` needed; key by URL or index).
  - `verdict` ← `claim['status']` (the new terminal status from 1.2).
- Fix `EvidenceLocker.tsx` typing: `claim_id: string`, `expandedClaims: string[]`, `toggleClaim(claimId: string)`.
- Add `getVerdictColor` cases for the actual statuses you'll emit (`verified`, `debunked`, `misleading`, `lacks_context`, `unverifiable`). Right now `misleading` and `lacks_context` are case-mismatched (component checks `'lacks context'` with a space, the enum is `'lacks_context'`).
- Render `content_snippet` under each source link in the expanded view — that's the missing "analysis" content.

---

## 3. Other issues found

### 3.1 Status enum is half-implemented

`ClaimStatus` defines 7 values. Only `PENDING`, `RESEARCHING`, and `UNVERIFIABLE` are ever set. `VERIFIED`, `DEBUNKED`, `MISLEADING`, `LACKS_CONTEXT` exist solely in routing functions that never trigger because the research stage doesn't adjudicate. After 1.2 is done, this gets resolved.

### 3.2 Inconsistent async wrapping

[`adversarial_review.py:594-608`](Backend-Agent/src/veritas_ai/nodes/adversarial_review.py#L594-L608) does an elaborate event-loop dance with `concurrent.futures.ThreadPoolExecutor` to handle "loop already running" cases. The other `_sync` wrappers just call `asyncio.run`. Standardize on one pattern (probably the simpler one — `asyncio.run` in a fresh thread is fine for the LangGraph synchronous streaming path).

### 3.3 Routing duplication

[`autonomous_graph.py`](Backend-Agent/src/veritas_ai/core/autonomous_graph.py) defines its own routing functions (`route_after_ingestion`, `route_claims_analysis_autonomous`, etc.) that partially duplicate `graph.py`'s routing functions. Since you don't use the HITL graph at all (see 4.1), delete the routing functions in `graph.py` and keep only the autonomous ones.

### 3.4 `revision_count` semantics

`revision_count` is incremented inside the critique node only when `is_revision_needed` is true. But `route_after_generation_autonomous` also checks the same counter to decide whether to send to critique at all. This is correct but subtle — document it or move both checks to one place.

### 3.5 Hardcoded fixtures in production code

[`claim_identification.py:458-486`](Backend-Agent/src/veritas_ai/nodes/claim_identification.py#L458-L486) defines `create_test_claims` in the production module. Move to tests or delete.

### 3.6 `TargetSelector.tsx` is unreachable

[`TargetSelector.tsx`](Frontend/src/components/workbench/TargetSelector.tsx) (modal for choosing video vs comment analysis) is not imported anywhere. `WorkbenchPage` comments out the relevant handler. Delete the file.

### 3.7 `CollaborationHub` and parts of `FinalizeActions` are pure mock

- [`CollaborationHub.tsx`](Frontend/src/components/workbench/CollaborationHub.tsx) is a "Coming Soon" stub. If refinement isn't on the near roadmap, remove it from the dashboard. Keeping it dimmed-out adds visual clutter and signals an unfinished product.
- [`FinalizeActions.tsx`](Frontend/src/components/workbench/FinalizeActions.tsx) renders "Save Report" and "Share Case" buttons with no `onClick` handlers — they're decorative. Either wire them up or delete them. Keep "Copy Final Comment".

### 3.8 `requirements.txt` is unrelated to this project

The 140-line `requirements.txt` includes `torch`, `opencv-python`, `openai-whisper`, `transformers`, `numba`, `numpy`, `tweepy`, `fastapi`, `asyncpg`, `SQLAlchemy`, `alembic`, etc. — none of which the codebase imports. The actual dependency list lives in `requirements-prod.txt` (~15 packages) which is correct. Delete `requirements.txt` and rename `requirements-prod.txt` → `requirements.txt`. Also delete `requirements-dev.txt` if you're not running tests.

### 3.9 Apify limit comment mismatch

[`api_usage.py:64`](Backend-Agent/src/veritas_ai/utils/api_usage.py#L64) raises *"limit of 800 reached"* when `count >= 600`. Fix the threshold or fix the message — currently the guard fires 200 calls before the message says it should.

### 3.10 Model name to verify

Both `claim_identification.py` and `content_ingestion.py` default to `gemini-3.1-flash-lite-preview`. Recent commit history (`b79bd14 Update model version to current gemini-3.1-flash-lite-preview`) suggests this was deliberate, but verify the model ID is currently valid in the Google AI Studio / Vertex APIs you're using. If it 404s in production, all four LLM nodes silently fall back to error handlers and you'll see empty responses with no obvious cause.

### 3.11 Logging hygiene

`main.py:19` has a stray `print(debug)`. Several modules call `logging.basicConfig` (e.g. [`runner.py:17`](Backend-Agent/src/veritas_ai/api/runner.py#L17)) which can fight with the Flask app's logger. Consolidate logging configuration in one place (probably `app.py`).

---

## 4. Dead code to delete

These are not used by the autonomous workflow. Total: ~3,500 lines.

| Path | Lines | Reason |
|---|---:|---|
| [`Backend-Agent/src/veritas_ai/core/hitl.py`](Backend-Agent/src/veritas_ai/core/hitl.py) | 701 | Human-in-the-loop infra. Only imported by `human_interaction.py`, which is also dead. |
| [`Backend-Agent/src/veritas_ai/nodes/human_interaction.py`](Backend-Agent/src/veritas_ai/nodes/human_interaction.py) | 811 | Not imported anywhere except by `hitl.py`. The autonomous graph bypasses HITL entirely. |
| [`Backend-Agent/src/veritas_ai/core/security.py`](Backend-Agent/src/veritas_ai/core/security.py) | 1032 | Not imported anywhere outside its own module. |
| [`Backend-Agent/src/veritas_ai/core/visualization.py`](Backend-Agent/src/veritas_ai/core/visualization.py) | 271 | Re-exported from `core/__init__.py` but never called. |
| Most of [`core/serialization.py`](Backend-Agent/src/veritas_ai/core/serialization.py) | ~380 of 399 | Only `safe_update_state` is used (in `graph.py`). Inline it or keep one trimmed file. |
| Routing fns in [`graph.py`](Backend-Agent/src/veritas_ai/core/graph.py): `route_content_processing`, `route_claims_analysis`, `route_after_research`, `route_after_generation`, `route_after_critique`, `route_after_human_interaction`, `setup_complete_veritas_workflow`, `add_complete_workflow_edges`, `create_basic_veritas_graph`, `create_minimal_graph` | ~250 | Replaced by autonomous-graph routing. |
| `MinimalGraphState`, `ChatMessage`, `chat_history` field + `add_messages` annotation in [`state.py`](Backend-Agent/src/veritas_ai/core/state.py) | ~30 | HITL artifacts. |
| `create_test_claims` in `claim_identification.py` | ~30 | Test fixture in prod code. |
| [`Frontend/src/components/workbench/TargetSelector.tsx`](Frontend/src/components/workbench/TargetSelector.tsx) | 124 | Unreachable, replaced by `WorkbenchPage`'s direct URL submit. |
| [`Backend-Agent/requirements.txt`](Backend-Agent/requirements.txt) | 140 | Bloated with unused libs (torch, opencv, etc.). Replace with `requirements-prod.txt`. |
| `Backend-Agent/test_audit.log`, `tests/integration/outputs/*.txt` | — | Stale dev byproducts. |
| `Backend-Agent/runtime.txt`, `Backend-Agent/Procfile` | — | Heroku artifacts; only keep if Heroku is the target. |

After this pass, the entire `core/` directory should be: `state.py`, `validation.py` (trimmed), `graph.py` (trimmed), `autonomous_graph.py`. The `nodes/` directory should be: `content_ingestion.py`, `claim_identification.py`, `evidence_research.py` (rewritten per 1.2), `response_generation.py`, `adversarial_review.py` (rewritten per 1.3 — possibly split into `critique/` sub-package).

---

## 5. Suggested order of attack

This is the order I'd actually do these in, optimizing for visible-improvement-per-hour:

1. **Fix `transform_state_for_frontend`** (2.3 backend half). Tiny change, immediately unblocks the Evidence Locker's analysis text and source list. ~30 min.
2. **Delete dead code** (section 4). Cathartic, makes the repo navigable, no behavior change. ~1 hr.
3. **Wire up `TargetDisplay`** with real metadata (2.1). Plumbing change in `transform_state_for_frontend` + the component. ~30 min.
4. **Rewrite `EvidenceResearcher` as a tool-using agent** (1.2). This is the biggest behavior win in the whole project — claims actually get adjudicated, the Evidence Locker shows real verdicts, and the verdict chip colors stop being meaningless. ~half a day.
5. **Pass structured critique data to the frontend** and rebuild `RedTeamLog` to show it (2.2). Now the panel says something. ~2 hrs.
6. **Rewrite `critique_response` as a sub-graph with web search** (1.3). Genuinely red-teams the response. ~half a day.
7. **Prompt cleanup pass** (1.4) — easy to do continuously while doing the above; do a final sweep at the end.
8. **Loosen claim extraction** (1.1) — small prompt + filter change, do it after 1.2 is in so you can see the deeper stack actually handle more claims.

After step 4 your demo is qualitatively different. After step 6 it's hard to distinguish from a much more expensive product.

---

## 6. Things I'd push back on

A few items I'd flag before changing:

- **Don't add a 15-claim hard cap if you're cost-sensitive.** Each claim is now ~3+ Tavily calls × 1 Gemini call. If the cap from 1.1 is 15, an expensive video could be 60+ tool calls. Either keep the cap modest (5–8) or add a per-session cost ceiling.
- **Don't remove `transcript_cleaner` blindly.** I suggested it might be unnecessary, but verify with a sample of raw Apify outputs first — TikTok captions are sometimes auto-generated noise that benefits from a cleanup pass.
- **Don't conflate `verification_summary` and `evidence_summary`.** They're semantically different (verdict vs. raw findings). Keep both fields when you fix the API utils, and decide in the UI which to surface where.
- **HITL deletion is one-way.** If you ever plan to re-add the collaboration hub, the `hitl.py` skeleton — though dead — is a non-trivial amount of design work. Consider stashing it on a branch instead of deleting outright.
