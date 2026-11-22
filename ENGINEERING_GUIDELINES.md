# CloudServe Engineering Guidelines & Architecture Standards

These engineering guidelines establish the non-negotiable standards, architecture principles, and coding protocols for human engineers and AI coding agents working on the CloudServe AI Support System.

---

## 1. Architectural North Star

CloudServe is a production-minded AI support platform designed to combine the flexibility of Large Language Models (LLMs) with the strict reliability, auditability, and safety guarantees of deterministic software engineering.

**Core Philosophy:**  
> *Probabilistic models inform decisions; deterministic software enforces rules.*

---

## 2. Inviolable Engineering Rules

### Rule 1: Deterministic Supremacy Over Probabilistic Outputs (A5, A6)
- Under no circumstances may an LLM output, generated response, or confidence score downgrade or bypass a mandatory escalation rule, safety block, or compliance trigger.
- Hard rules (outage detection, data breach reports, legal threats, prompt injection patterns, `must_not_auto_respond` flags) are evaluated in deterministic code. Their outcomes are final.

### Rule 2: Zero Citation Invention & Verification (A4, A6)
- The system must never fabricate, extrapolate, or approximate a knowledge-base citation.
- Every citation emitted in an automated answer must match the exact `doc_id` and section title of a chunk retrieved from the 29 official knowledge-base articles in `data/kb/documentation.json`.
- Any answer containing an ungrounded or hallucinated citation must be intercepted by post-generation guardrails and degraded to `escalate`.

### Rule 3: Preservation of All 22 Intent Categories & 3 Urgency Tiers (A3)
- Do not reduce or collapse the 22 dataset intent categories into broad groups:
  `account_access`, `api_key_issue`, `api_usage_question`, `authentication_failure`, `billing_query`, `compliance_request`, `configuration_help`, `data_export`, `data_residency`, `database_issue`, `deployment_failure`, `feature_request`, `integration_help`, `onboarding`, `performance_degradation`, `quota_or_overage`, `rate_limit`, `rollback_request`, `security_incident`, `sso_configuration`, `unclear_request`, `webhook_issue`.
- Urgency tiers must strictly match the dataset schema: `low`, `medium`, `high`.
- Channels must strictly match the dataset schema: `email`, `chat`, `docs_comment`, `forum`.

### Rule 4: Active Guardrail Blocking (A7)
- The system must implement active safety guardrails that detect private data (PII), adversarial inputs (prompt injection, jailbreak), and unsupported claims, actively blocking the response.
- Blocked tickets must record the triggered guardrail in the audit log and halt generation immediately.

### Rule 5: Strict Data Isolation (Zero Ground-Truth Leakage) (A9)
- `data/val/ground_truth_responses.json` and validation datasets are strictly reserved for the evaluation pipeline (`scripts/evaluate.py`).
- Runtime code (`src/cloudserve/`) must **NEVER** import, read, or have access to `ground_truth_responses.json` or any expected evaluation answers.

### Rule 6: Complete Decision Reconstruction in Audit Logs (A8)
- Every automated decision must be persistently logged to SQLite (via synchronous SQLAlchemy) with full fidelity.
- Decisions logged must reconcile 1:1 with tickets processed.
- The stored audit record must capture:
  `decision_id`, `created_at`, `ticket_id`, `stage`, `input_summary`, `model`, `prediction`, `confidence`, `alternatives`, `sources_used`, `threshold_applied`, `action_taken`, `reason`, `guardrail_results`, `prompt_version`, `requirement_ids`.
- A reviewer must be able to completely reconstruct how and why a decision was reached using only the database record.

### Rule 7: Resilience & Operational Failure Handling (A11)
- The system must handle operational failures gracefully without crashing:
  - LLM provider timeouts, 429 rate limits, and 5xx outages must employ exponential backoff with jitter and gracefully fallback to escalation.
  - Malformed ticket payloads must be caught at the API boundary with HTTP 422 validation errors.
  - Empty or low-similarity retrieval results must route tickets to escalation rather than hallucinating unsupported answers.

### Rule 8: Arbitrary I/O Paths for Unattended Evaluation (A9, A10)
- The evaluation runner (`scripts/evaluate.py`) must operate without human intervention and accept arbitrary input paths (`--input`) and output paths (`--output`).
- Reports must automatically output across all three tiers:
  - **Tier 1 (Business):** First Contact Resolution (>=60%), median reply time (<5min), escalation rate (<=30%).
  - **Tier 2 (Technical):** Per-class precision & recall (>=85%), hallucination rate (<=5%), citation accuracy (>=95%), latency p95 (<3s).
  - **Tier 3 (Governance):** Zero PII occurrences, calibration error within 5 points of observed accuracy, cross-group variation (<5 percentage points).

### Rule 9: Simplicity First in Persistence
- Use **synchronous SQLAlchemy** with SQLite (`decisions.db`).
- Do not introduce asynchronous database drivers (`aiosqlite`) or distributed databases unless high-concurrency profiling demonstrates that synchronous operations are a proven bottleneck.

### Rule 10: Single Documented Test Command (A12)
- The entire test suite must execute cleanly using a single command: `pytest`.
- Never claim a component or acceptance criterion has passed without executing and verifying its tests in the terminal.

---

## 3. Directory Isolation & Ownership

| Directory | Ownership & Access Boundary | Allowed Dependencies |
| :--- | :--- | :--- |
| `src/cloudserve/` | Core application and runtime logic | Standard libraries, declared dependencies, `prompts/` |
| `data/kb/` | Static knowledge base (`documentation.json`, 29 articles) | Read-only input to indexer |
| `data/dev/` | Development tickets (`development_tickets.json`, 500 tickets) | Developer experiments, training/tuning |
| `data/val/` | Validation tickets & `ground_truth_responses.json` | **ONLY** `scripts/evaluate.py` and evaluation tests |
| `data/runtime/` | SQLite database & Chroma DB directory | Ignored in Git; written at runtime |
| `prompts/` | Versioned prompt templates | Read by `src/cloudserve` prompt loaders |
| `scripts/` | CLI tools (`index_kb.py`, `evaluate.py`, `run_server.py`) | Invokes `src/cloudserve` and data directories |
| `tests/` | Unit, integration, resilience, and eval tests | Imports `src/cloudserve` and test fixtures |

---

## 4. Benchmark & Metrics Standards

The system evaluation and reporting must preserve the distinction between the Evaluation Framework baselines, development dataset historical outcomes, and project targets:

| Metric | Source A: Evaluation Framework Baseline<br>*(Target Benchmark in `Evaluation_Framework.docx`)* | Source B: Development Dataset Historical Outcomes<br>*(Calculated from `data/dev/development_tickets.json`)* | Target for CloudServe<br>*(Success Threshold)* |
| :--- | :-: | :-: | :-: |
| **First Contact Resolution (FCR)** | **42.0%** | **43.8%** (219 / 500 records where `history.first_contact_resolution == true`) | **$\ge$ 60.0%** |
| **Escalation Rate** | **58.0%** | **56.2%** (281 / 500 records where `history.escalated == true`) | **$\le$ 30.0%** |
| **Response / Resolution Times** | 8 to 12 hours *(first-response time)* | Mean: **421.7 min**, Median: **214.0 min** *(resolution time via `history.resolution_time_minutes`)* | **Median First Reply < 5 min** |
| **Customer Satisfaction (CSAT)** | 3.2 / 5.0 | Mean: **2.97 / 5.0** (2.972 from `history.csat_rating`) | **$\ge$ 4.0 / 5.0** |
| **Repeat Contacts** | *"Not measured"* | **21.6%** (108 / 500 records where `history.repeat_contact == true`) | **Halved ($\le$ 10.8%)** |

> *Crucial Metric Distinction:* `history.resolution_time_minutes` measures total **resolution time**, not first-response time. First-response time is an external organizational baseline (8–12 hours) from the Project Brief.

---

## 5. Acceptance Criteria Status Protocol

- Under no circumstances may any row in `docs/traceability_matrix.md` or `docs/acceptance_criteria.md` be marked as `Passed` or `Complete` until:
  1. The code is implemented.
  2. The automated tests in `tests/` are written.
  3. The test command `pytest` has been executed in the shell and passed with zero failures.
