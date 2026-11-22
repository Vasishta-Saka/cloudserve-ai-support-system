# CloudServe AI Customer Support System
## Explicit First Commit Manifest & Publication Rights Audit

**Document ID:** `MANIFEST-FIRST-COMMIT-2026-11-22`  
**Date:** November 22, 2026  
**Auditor / Release Engineer:** Senior DevSecOps & Platform Engineer  
**Program:** IIT Roorkee — Post Graduate Program in Forward Deployed AI Engineering  
**Repository Working Directory:** `C:\Projects\CloudServe-AI-Support-System`  
**Git Working Tree Target:** `master` (Unborn baseline)  
**Total Candidate Paths:** 102 files  

---

## 1. Classification Methodology

In strict compliance with DevSecOps release standards and course guidelines, every repository file is evaluated against four criteria:
1. **Source & Functionality:** Required runtime code, build configuration, automated tests, versioned prompts, and documentation.
2. **Security & PII:** Zero secrets, credentials, API keys, or personal identifiable customer information.
3. **Intellectual Property & Authorship:** Authored by the project candidate or team.
4. **Publication & Redistribution Rights:** Confirmed legal right to redistribute publicly.
   - Files with clear candidate authorship and zero third-party copyright $\rightarrow$ **`INCLUDE`**
   - Files originating from course benchmark packs where public redistribution rights have not been formally confirmed in writing $\rightarrow$ **`REVIEW REQUIRED`**
   - Files containing runtime state, private evaluation runs, student submission workbooks, or secrets $\rightarrow$ **`EXCLUDE`** (Strictly ignored by `.gitignore`)

---

## 2. Path-by-Path Staging Manifest

| # | File Relative Path | Size (Bytes) | Classification | Subsystem / Description | Publication Rights Notes |
| :---: | :--- | :---: | :---: | :--- | :--- |
| 1 | `.env.example` | 1,530 | **INCLUDE** | Configuration | Template environment variables with placeholder keys only. Safe for public release. |
| 2 | `.github/workflows/ci.yml` | 997 | **INCLUDE** | CI Pipeline | Multi-version Python CI (3.10, 3.11, 3.12). Runs lint, type-check, KB index, and pytest. |
| 3 | `.gitignore` | 1,671 | **INCLUDE** | Repository Hygiene | Comprehensive exclusion rules for venvs, runtime DBs, caches, submission workbooks. |
| 4 | `ENGINEERING_GUIDELINES.md` | 8,308 | **INCLUDE** | Documentation | Architectural principles, deterministic supremacy rules, and engineering standards. |
| 5 | `README.md` | 33,643 | **INCLUDE** | Documentation | Master runbook, clean-room reproducibility guide, architecture diagrams. |
| 6 | `pyproject.toml` | 1,609 | **INCLUDE** | Packaging | Modern build specification (PEP 518/621). MIT declaration cleanly removed. |
| 7 | `requirements.txt` | 903 | **INCLUDE** | Dependencies | Pinned production runtime dependencies. |
| 8 | `requirements-dev.txt` | 465 | **INCLUDE** | Dependencies | Development and test dependencies (pytest, pytest-cov, ruff, mypy). |
| 9 | `docs/acceptance_criteria.md` | 5,132 | **INCLUDE** | Documentation | Formal mapping of acceptance criteria A1 through A12. Authored by candidate. |
| 10 | `docs/architecture.md` | 10,400 | **INCLUDE** | Documentation | 5-gate pipeline specification and routing state machine documentation. |
| 11 | `docs/traceability_matrix.md` | 4,491 | **INCLUDE** | Documentation | Test-to-specification traceability matrix. Authored by candidate. |
| 12 | `reports/github/CLEAN_ROOM_VERIFICATION_REPORT.md` | 11,397 | **INCLUDE** | Engineering Audit | Phase 2 clean-room verification traces and exit codes. |
| 13 | `reports/github/GITHUB_READINESS_AUDIT.md` | 12,341 | **INCLUDE** | Engineering Audit | Phase 1 repository portability and structure audit. |
| 14 | `reports/github/FINAL_RELEASE_READINESS_AUDIT.md` | 23,677 | **INCLUDE** | Engineering Audit | Forensic review of clean-room reproduction evidence and operational limits. |
| 15 | `reports/github/PHASE_3_GIT_HYGIENE_REMEDIATION.md` | 17,807 | **INCLUDE** | Engineering Audit | Phase 3 Git hygiene, .gitignore hardening, and README updates report. |
| 16 | `reports/github/FINAL_STAGING_READINESS_REPORT.md` | 28,348 | **INCLUDE** | Engineering Audit | Phase 4 Git history investigation and license verification report. |
| 17 | `reports/github/PHASE_4_1_PRE_STAGING_BLOCKER_RESOLUTION.md` | *Pending* | **INCLUDE** | Engineering Audit | Phase 4.1 Ruff resolution, CI verification, and publication rights review. |
| 18 | `reports/github/EXPLICIT_FIRST_COMMIT_MANIFEST.md` | *This file* | **INCLUDE** | Engineering Audit | Comprehensive path-by-path staging manifest and publication rights audit. |
| 19 | `data/dev/.gitkeep` | 73 | **INCLUDE** | Dataset Structure | Directory retention marker for development data. |
| 20 | `data/dev/manual_smoke_1.json` | 903 | **INCLUDE** | Test Fixture | 1-ticket smoke test fixture authored by candidate team for pre-flight testing. |
| 21 | `data/dev/manual_smoke_5.json` | 4,720 | **INCLUDE** | Test Fixture | 5-ticket smoke test fixture authored by candidate team for pre-flight testing. |
| 22 | `data/kb/.gitkeep` | 69 | **INCLUDE** | Dataset Structure | Directory retention marker for knowledge-base data. |
| 23 | `data/val/.gitkeep` | 132 | **INCLUDE** | Dataset Structure | Directory retention marker for validation data. |
| 24 | `data/kb/documentation.json` | 40,771 | **REVIEW REQUIRED** | Benchmark KB | 29 synthetic support articles supplied by IIT Roorkee capstone program. |
| 25 | `data/dev/development_tickets.json` | 453,003 | **REVIEW REQUIRED** | Benchmark Data | 500 synthetic development tickets supplied by IIT Roorkee capstone program. |
| 26 | `data/val/validation_tickets.json` | 71,999 | **REVIEW REQUIRED** | Benchmark Data | 80 synthetic validation tickets supplied by IIT Roorkee capstone program. |
| 27 | `data/val/ground_truth_responses.json` | 221,693 | **REVIEW REQUIRED** | Benchmark Data | 200 synthetic reference responses supplied by IIT Roorkee capstone program. |
| 28 | `prompts/classification/.gitkeep` | 97 | **INCLUDE** | Prompt Governance | Directory retention marker. |
| 29 | `prompts/classification/README.md` | 689 | **INCLUDE** | Prompt Governance | Version history and change log for classification system prompts. |
| 30 | `prompts/classification/v1.txt` | 8,807 | **INCLUDE** | Prompts | Version 1 classification prompt baseline. |
| 31 | `prompts/classification/v2.txt` | 10,546 | **INCLUDE** | Prompts | Version 2 PRD-hardened classification prompt with calibrated urgency criteria. |
| 32 | `prompts/generation/.gitkeep` | 81 | **INCLUDE** | Prompt Governance | Directory retention marker. |
| 33 | `prompts/guardrails/.gitkeep` | 93 | **INCLUDE** | Prompt Governance | Directory retention marker. |
| 34 | `prompts/response_generation/v1.txt` | 8,235 | **INCLUDE** | Prompts | Grounded response generation prompt template. |
| 35 | `scripts/.gitkeep` | 78 | **INCLUDE** | Scripts Structure | Directory retention marker. |
| 36 | `scripts/evaluate.py` | 8,194 | **INCLUDE** | CLI Harness | Unattended evaluation CLI runner supporting mock and live execution modes. |
| 37 | `scripts/index_kb.py` | 4,150 | **INCLUDE** | CLI Utility | Vector indexing CLI tool for populating ChromaDB knowledge-base store. |
| 38 | `scripts/run_targeted_security_regression.py` | 4,670 | **INCLUDE** | Test Script | Targeted security test harness for prompt injection and security escalations. |
| 39 | `src/cloudserve/__init__.py` | 510 | **INCLUDE** | Source: Core | Package initialization and top-level exports. |
| 40 | `src/cloudserve/api/__init__.py` | 79 | **INCLUDE** | Source: API | API package initialization. |
| 41 | `src/cloudserve/api/routes/__init__.py` | 84 | **INCLUDE** | Source: API | API routing subpackage. |
| 42 | `src/cloudserve/classification/__init__.py` | 421 | **INCLUDE** | Source: Classifier | Classifier exports. |
| 43 | `src/cloudserve/classification/classifier.py` | 8,601 | **INCLUDE** | Source: Classifier | Hybrid multi-class intent and operational urgency classifier. |
| 44 | `src/cloudserve/classification/models.py` | 4,026 | **INCLUDE** | Source: Classifier | Pydantic domain models for classification inputs and outputs. |
| 45 | `src/cloudserve/classification/prompts.py` | 1,486 | **INCLUDE** | Source: Classifier | Dynamic prompt loading utilities. |
| 46 | `src/cloudserve/config.py` | 7,085 | **INCLUDE** | Source: Config | Global settings via `pydantic-settings`. |
| 47 | `src/cloudserve/evaluation/__init__.py` | 1,238 | **INCLUDE** | Source: Evaluation | Evaluation package exports. |
| 48 | `src/cloudserve/evaluation/answer_eval.py` | 24,210 | **INCLUDE** | Source: Evaluation | Ground-truth answer quality evaluator (`must_mention`, `must_not_claim`). |
| 49 | `src/cloudserve/evaluation/factory.py` | 16,374 | **INCLUDE** | Source: Evaluation | Pipeline factory for mock and live evaluation runners. |
| 50 | `src/cloudserve/evaluation/human_review.py` | 16,613 | **INCLUDE** | Source: Evaluation | Stratified sampling and Cohen's Kappa reconciliation calculator. |
| 51 | `src/cloudserve/evaluation/metrics.py` | 18,164 | **INCLUDE** | Source: Evaluation | Accuracy, latency, F1, and routing compliance metric evaluators. |
| 52 | `src/cloudserve/evaluation/policy_eval.py` | 28,332 | **INCLUDE** | Source: Evaluation | Deterministic routing policy compliance verifier. |
| 53 | `src/cloudserve/evaluation/reporter.py` | 48,241 | **INCLUDE** | Source: Evaluation | Markdown and JSON evaluation report generator. |
| 54 | `src/cloudserve/evaluation/retrieval_eval.py` | 13,382 | **INCLUDE** | Source: Evaluation | Precision, Recall, and MRR retrieval metric calculators. |
| 55 | `src/cloudserve/evaluation/runner.py` | 30,923 | **INCLUDE** | Source: Evaluation | Batch execution engine with atomic checkpointing and resumability. |
| 56 | `src/cloudserve/generation/__init__.py` | 1,327 | **INCLUDE** | Source: Generation | Generation package exports. |
| 57 | `src/cloudserve/generation/grounding.py` | 44,775 | **INCLUDE** | Source: Generation | Sentence-level citation extraction, NLI grounding verifier, and hallucination shield. |
| 58 | `src/cloudserve/generation/llm_client.py` | 19,332 | **INCLUDE** | Source: Generation | Resilient Groq provider client with exponential backoff and mock offline adapter. |
| 59 | `src/cloudserve/generation/models.py` | 5,553 | **INCLUDE** | Source: Generation | Pydantic data models for candidate drafts and grounding results. |
| 60 | `src/cloudserve/generation/prompts.py` | 6,701 | **INCLUDE** | Source: Generation | Prompt interpolation and XML passage packing utilities. |
| 61 | `src/cloudserve/generation/response_generator.py` | 15,517 | **INCLUDE** | Source: Generation | Grounded response generation coordinator. |
| 62 | `src/cloudserve/guardrails/__init__.py` | 203 | **INCLUDE** | Source: Guardrails | Guardrail package exports. |
| 63 | `src/cloudserve/guardrails/safety.py` | 9,385 | **INCLUDE** | Source: Guardrails | Regex-based PII redaction shield and prompt injection attack detector. |
| 64 | `src/cloudserve/ingestion/__init__.py` | 977 | **INCLUDE** | Source: Ingestion | Ingestion package exports. |
| 65 | `src/cloudserve/ingestion/models.py` | 6,961 | **INCLUDE** | Source: Ingestion | Unified ticket schema normalizing email, chat, forum, and documentation comments. |
| 66 | `src/cloudserve/ingestion/parsers.py` | 12,536 | **INCLUDE** | Source: Ingestion | Channel-specific normalizers and metadata sanitizers. |
| 67 | `src/cloudserve/rag/__init__.py` | 982 | **INCLUDE** | Source: RAG | Retrieval package exports. |
| 68 | `src/cloudserve/rag/chunker.py` | 7,631 | **INCLUDE** | Source: RAG | Section-aware and fixed-size knowledge base article chunker. |
| 69 | `src/cloudserve/rag/indexer.py` | 9,765 | **INCLUDE** | Source: RAG | ChromaDB collection indexer supporting local embeddings and mock feature hashing. |
| 70 | `src/cloudserve/rag/models.py` | 4,547 | **INCLUDE** | Source: RAG | Models for KB articles, chunks, and retrieved passages. |
| 71 | `src/cloudserve/rag/retriever.py` | 8,008 | **INCLUDE** | Source: RAG | ChromaDB passage retriever with relevance scoring and threshold filtering. |
| 72 | `src/cloudserve/routing/__init__.py` | 1,754 | **INCLUDE** | Source: Routing | Routing package exports. |
| 73 | `src/cloudserve/routing/billing.py` | 6,168 | **INCLUDE** | Source: Routing | Billing risk evaluator and disputed charge detector. |
| 74 | `src/cloudserve/routing/citations.py` | 3,947 | **INCLUDE** | Source: Routing | Citation format validator and document ID verifier. |
| 75 | `src/cloudserve/routing/models.py` | 8,765 | **INCLUDE** | Source: Routing | Deterministic routing outcomes and standardized reason codes. |
| 76 | `src/cloudserve/routing/router.py` | 46,625 | **INCLUDE** | Source: Routing | Deterministic routing engine implementing PRD v2 precedence rules. |
| 77 | `src/cloudserve/service.py` | 49,822 | **INCLUDE** | Source: Service | Master `TicketProcessor` orchestrating the 5-gate support pipeline. |
| 78 | `src/cloudserve/storage/__init__.py` | 738 | **INCLUDE** | Source: Storage | Persistence package exports. |
| 79 | `src/cloudserve/storage/database.py` | 5,020 | **INCLUDE** | Source: Storage | SQLite database engine configuration and thread-safe session management. |
| 80 | `src/cloudserve/storage/models.py` | 27,771 | **INCLUDE** | Source: Storage | SQLAlchemy 2.0 ORM models for decision audit logs and processing runs. |
| 81 | `src/cloudserve/storage/repository.py` | 34,370 | **INCLUDE** | Source: Storage | Append-only decision audit repository ensuring 100% ticket reconciliation. |
| 82 | `tests/__init__.py` | 59 | **INCLUDE** | Test Suite | Test package root. |
| 83 | `tests/conftest.py` | 306 | **INCLUDE** | Test Suite | Pytest shared fixtures and mock environment initializers. |
| 84 | `tests/evaluation/__init__.py` | 88 | **INCLUDE** | Test Suite | Evaluation test subpackage. |
| 85 | `tests/integration/__init__.py` | 85 | **INCLUDE** | Test Suite | Integration test subpackage. |
| 86 | `tests/unit/__init__.py` | 49 | **INCLUDE** | Test Suite | Unit test subpackage. |
| 87 | `tests/unit/test_answer_quality_eval.py` | 18,951 | **INCLUDE** | Unit Tests | Tests for answer quality evaluators and reference matching. |
| 88 | `tests/unit/test_classification.py` | 30,737 | **INCLUDE** | Unit Tests | Tests for intent classification, prompt loading, and urgency scoring. |
| 89 | `tests/unit/test_config.py` | 3,944 | **INCLUDE** | Unit Tests | Tests for environment settings and path resolution. |
| 90 | `tests/unit/test_evaluation_metrics_and_policy.py` | 47,142 | **INCLUDE** | Unit Tests | Tests for metric calculators and routing policy compliance. |
| 91 | `tests/unit/test_evaluation_resumability.py` | 13,926 | **INCLUDE** | Unit Tests | Tests for checkpointing, dataset fingerprinting, and run resumability. |
| 92 | `tests/unit/test_evaluation_runner.py` | 30,640 | **INCLUDE** | Unit Tests | Tests for batch evaluation runner CLI and error handling. |
| 93 | `tests/unit/test_generation.py` | 168,608 | **INCLUDE** | Unit Tests | Tests for candidate generation, grounding validation, and regression cases. |
| 94 | `tests/unit/test_ingestion.py` | 20,318 | **INCLUDE** | Unit Tests | Tests for ticket parsing, whitespace normalization, and channel models. |
| 95 | `tests/unit/test_llm_client.py` | 26,988 | **INCLUDE** | Unit Tests | Tests for Groq client resilience, retry backoff, and mock fallback. |
| 96 | `tests/unit/test_orchestration.py` | 52,547 | **INCLUDE** | Unit Tests | Tests for end-to-end 5-gate pipeline integration. |
| 97 | `tests/unit/test_persistence.py` | 97,344 | **INCLUDE** | Unit Tests | Tests for SQLite audit logging, schema migration, and data minimization. |
| 98 | `tests/unit/test_policy_parity.py` | 5,207 | **INCLUDE** | Unit Tests | Tests for deterministic routing parity across equivalent scenarios. |
| 99 | `tests/unit/test_rag.py` | 23,210 | **INCLUDE** | Unit Tests | Tests for KB chunking, indexing, and Chroma passage retrieval. |
| 100 | `tests/unit/test_routing.py` | 74,645 | **INCLUDE** | Unit Tests | Tests for PRD v2 routing precedence, billing gates, and escalation logic. |
| 101 | `tests/unit/test_security_classification_regression.py` | 5,938 | **INCLUDE** | Unit Tests | Regression tests for prompt injection and security escalation triggers. |
| 102 | `tests/unit/test_urgency_calibration.py` | 9,570 | **INCLUDE** | Unit Tests | Tests for operational urgency classification thresholds. |

---

## 3. Publication Rights Review & Outstanding Questions

### 3.1 Course-Provided Benchmark Datasets (`REVIEW REQUIRED` - 4 Files)
The following 4 files originate from the academic course distribution pack:
- `data/kb/documentation.json` (29 simulated support articles)
- `data/dev/development_tickets.json` (500 benchmark development tickets)
- `data/val/validation_tickets.json` (80 benchmark validation tickets)
- `data/val/ground_truth_responses.json` (200 reference response mappings)

**Diligent Legal Finding:**
- All 4 files are 100% synthetic and contain zero personal identifiers or commercial credentials.
- However, they were provided by IIT Roorkee / course instructors as part of the capstone evaluation pack.
- Because no explicit written license granting public redistribution was located in the course documentation (`docs/reference/`), publishing these files on a public GitHub release carries potential course policy or copyright questions.
- **Release Decision:** In strict accordance with Phase 4.1 instructions (*"Where rights are unclear, use REVIEW REQUIRED and do not stage the file"*), these 4 files are marked `REVIEW REQUIRED`.
- **Options for the User:**
  1. **Option A (Include All):** If the candidate confirms that IIT Roorkee permits public posting of synthetic capstone benchmark datasets, all 102 files can be staged.
  2. **Option B (Code-Only Release):** If course datasets must remain private, stage the 98 candidate-authored files (`INCLUDE`) and add `data/dev/development_tickets.json`, `data/val/*.json`, and `data/kb/documentation.json` to `.gitignore`. Users can populate their own data or use `manual_smoke_1.json` and `manual_smoke_5.json`.

---

## 4. Reconciled Manifest Summary

```
=========================================================================================================
                                     STAGING RECONCILIATION SUMMARY
=========================================================================================================
Classification Category       File Count   Percentage     Description / Action Required
---------------------------------------------------------------------------------------------------------
INCLUDE                       98 files     96.08%         Approved candidate-authored code, tests, docs
REVIEW REQUIRED                4 files      3.92%         Course-provided benchmark datasets (Awaiting confirmation)
EXCLUDE                      Hundreds      N/A            Runtime DBs, venvs, submission workbooks, raw evals
---------------------------------------------------------------------------------------------------------
TOTAL UNTRACKED FILES        102 files    100.00%         Matches `git status --short --untracked-files=all`
=========================================================================================================
```

*Manifest certified by Repository Release Engineer — CloudServe AI Project*
