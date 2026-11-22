# CloudServe AI Support System - Traceability Matrix

This matrix maps each official acceptance criterion from `docs/reference/Build_Specification.docx` (Section 02) to its component requirement, source module, automated test, status, and evidence of completion.

> **Status Notice:** No acceptance criterion may be marked as passed until its implementation has been executed and verified through automated tests.

---

## Traceability Mapping

| Acceptance Criterion | PRD / Component Requirement | Source Module | Automated Test File | Status | Evidence of Completion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A1**<br>Clean installation and startup | Clean install, virtualenv setup, and startup following `README.md` | `pyproject.toml`, `README.md`, `src/cloudserve/api/app.py` | `tests/integration/test_startup.py` | Pending | Pending implementation & test execution |
| **A2**<br>Ingestion of all 4 channels | Ingestion of `email`, `chat`, `docs_comment`, `forum` into single normalized internal model | `src/cloudserve/ingestion/` | `tests/unit/test_ingestion.py` | Pending | Pending implementation & test execution |
| **A3**<br>Intent, urgency & numeric confidence | Classification across all 22 intent classes, 3 urgency tiers (`low`, `medium`, `high`), calibrated float confidence `[0.0, 1.0]`, fallback handling, alternative tracking | `src/cloudserve/classification/` | `tests/unit/test_classification.py` | Pending | Pending implementation & test execution |
| **A4**<br>Ranked & identifiable KB passages | Vector retrieval over 29 KB articles (`data/kb/documentation.json`) via Chroma & `all-MiniLM-L6-v2`; score thresholding; identifiable source IDs | `src/cloudserve/rag/` | `tests/unit/test_rag_retrieval.py` | Pending | Pending implementation & test execution |
| **A5**<br>Deterministic routing | Routing decision between `auto_respond` and `escalate` using data-driven threshold; identical inputs yield identical outputs; support-readable reason | `src/cloudserve/routing/` | `tests/unit/test_deterministic_routing.py` | Pending | Pending implementation & test execution |
| **A6**<br>Grounded responses with valid citations | Response grounded strictly in retrieved passages; citations attached to claims resolving to real KB passages; no hallucinated citations | `src/cloudserve/generation/`, `src/cloudserve/rag/citation.py` | `tests/unit/test_citation_verifier.py` | Pending | Pending implementation & test execution |
| **A7**<br>At least one blocking guardrail | Active guardrail checking for private data (PII) and unsupported claims; actively blocks response when triggered | `src/cloudserve/guardrails/safety.py` | `tests/unit/test_guardrails.py` | Pending | Pending implementation & test execution |
| **A8**<br>Persistent decision logging | Full persistent audit log in SQLite containing required schema (decision_id, timestamp, ticket_id, stage, model, prediction, alternatives, sources_used, threshold, action_taken, reason, guardrail_results, prompt_version, requirement_ids); 1:1 reconciliation with processed tickets | `src/cloudserve/storage/` | `tests/integration/test_storage_audit.py` | Pending | Pending implementation & test execution |
| **A9**<br>Unattended evaluation run | Standalone unattended run over full evaluation set; no manual intervention; accepts arbitrary `--input` and `--output` paths | `scripts/evaluate.py`, `src/cloudserve/evaluation/` | `tests/evaluation/test_eval_pipeline.py` | Pending | Pending implementation & test execution |
| **A10**<br>Automatic metrics report generation | Automatic calculation and reporting of business outcomes (FCR, reply time, escalation rate), technical performance (per-class P/R, latency p95), governance conditions (PII count, calibration, cross-group variation) | `src/cloudserve/evaluation/` | `tests/evaluation/test_metrics_reporting.py` | Pending | Pending implementation & test execution |
| **A11**<br>Failure handling without crashing | System handles missing retrieval hits, provider outages/timeouts, 429 rate limits, and malformed inputs gracefully without crashing | `src/cloudserve/api/`, `src/cloudserve/generation/` | `tests/integration/test_resilience.py` | Pending | Pending implementation & test execution |
| **A12**<br>One documented test command | Entire test suite executes via a single documented command (`pytest`) and passes with exit code 0 | `pyproject.toml`, `tests/` | Test runner invocation (`pytest`) | Pending | Pending implementation & test execution |
