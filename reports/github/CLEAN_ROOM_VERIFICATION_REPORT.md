# CloudServe AI Customer Support System
## Phase 2: Isolated Clean-Room Reproducibility Verification Report
**Program:** Post Graduate Certificate in Forward Deployed AI Engineering (IIT Roorkee)  
**Candidate:** Vasishta Saka  
**Source Repository:** `C:\Projects\CloudServe-AI-Support-System`  
**Isolated Clean-Room Directory:** `C:\Temp\CloudServe-CleanRoom-Test`  
**Date of Verification:** November 22, 2026  
**Auditor Roles:** Senior Python Platform Engineer, DevSecOps Engineer & Lead AI Architect  

---

## 1. Executive Summary & Final Verdict

A rigorous, end-to-end clean-room reproducibility audit was conducted in an isolated directory (`C:\Temp\CloudServe-CleanRoom-Test`) completely disconnected from the source repository's `.venv`, pre-existing database files, and cached vector stores.

### Verification Results Summary:
- **Clean Virtual Environment:** Fresh Python 3.12.4 virtual environment created and all dependencies successfully installed via `requirements.txt` and `requirements-dev.txt` (Exit code 0).
- **Working-Directory Independence:** Verified settings loading and dynamic prompt loading from repository root, subdirectories (`tests/`), and external directories (`C:\Temp`).
- **Knowledge-Base Indexing:** Fresh Chroma vector index generated from `data/kb/documentation.json`, indexing 29 articles into 116 section-aware chunks.
- **SQLite Database Initialization:** Parent directories auto-created, schemas auto-migrated, and full record persistence/read-back verified.
- **Application Startup & End-to-End Processing:** Ticket processing executed via `create_ticket_processor()`, correctly routing tickets and committing immutable audit logs.
- **Automated Test Suite:** **All 452 unit tests passed in 68.13s with 92% statement coverage** (Exit code 0).
- **Unattended Mock Smoke Evaluation:** Executed `scripts/evaluate.py --mode mock`, generating all 9 evaluation artifacts and 100% audit reconciliation.
- **Dataset Privacy & Licensing:** 100% synthetic, zero real customer PII or credentials detected across `data/dev/`, `data/val/`, and `data/kb/`.

### Final Reproducibility Verdict:
### **`FULLY REPRODUCIBLE`** (with documented working-directory convention)

---

## 2. Test 1: Fresh Environment & Dependency Installation

### 2.1 Execution Parameters:
- **Isolation Path:** `C:\Temp\CloudServe-CleanRoom-Test`
- **Python Version:** `Python 3.12.4 (tags/v3.12.4:8e8a4ba, Jun 6 2024, 19:30:16) [MSC v.1940 64 bit (AMD64)]`
- **Pip Upgrade Command:** `C:\Temp\CloudServe-CleanRoom-Test\.venv\Scripts\python.exe -m pip install --upgrade pip` (Upgraded to `pip 26.3`)
- **Package Installation Command:**  
  `pip install -r requirements.txt -r requirements-dev.txt`
- **Editable Install Command:**  
  `pip install -e .`

### 2.2 Results & Package Verification:
- **Exit Code:** `0` (Success).
- **Installed Packages:** 255 site-packages directories installed, including `fastapi (0.163.1)`, `chromadb (1.5.9)`, `sentence-transformers (6.3.0)`, `langchain-groq (1.2.1)`, `sqlalchemy (2.1.9)`, `pydantic (2.14.3)`, `pytest (9.2.0)`, `ruff (0.17.5)`, and `mypy (2.4.1)`.
- **Dependency Conflicts:** Zero version collisions or missing transitive dependencies detected.
- **Application Importability:** `from cloudserve import create_ticket_processor` succeeded with zero errors.

---

## 3. Test 2: Working-Directory Independence & Path Resolution

### 3.1 Investigation of Previous Working Directory Failure:
During initial probing, running Python from a subdirectory yielded `ModuleNotFoundError: No module named 'cloudserve'`.
- **Root Cause Analysis:** This was not an application configuration failure; it occurred because the package was tested before running `pip install -e .`. Once the package was installed in editable mode, `cloudserve` became globally importable across the environment.

### 3.2 Multi-Directory Configuration Testing:
Settings loading and prompt template loading were tested across three distinct working directories:
1. **Repository Root (`C:\Temp\CloudServe-CleanRoom-Test`):** Settings instantiated, prompt v2 loaded (10,359 characters).
2. **Subdirectory (`C:\Temp\CloudServe-CleanRoom-Test\tests`):** Settings instantiated, prompt v2 loaded (10,359 characters).
3. **External Directory (`C:\Temp`):** Settings instantiated, prompt v2 loaded (10,359 characters).

### 3.3 Path Resolution Architecture:
- **Prompt Files:** Resolved via `Path(__file__).resolve().parents[3] / "prompts" / ...`, guaranteeing 100% working-directory independence.
- **Database & RAG Defaults:** `src/cloudserve/config.py` defaults to `./data/runtime/cloudserve.db` and `./data/runtime/chroma_db`. When executing CLI scripts (`scripts/evaluate.py`, `scripts/index_kb.py`), the command must be launched from the repository root, or paths must be set via environment variables (`CHROMA_PERSIST_DIRECTORY`, `DATABASE_URL`). This is explicitly documented in `README.md`.

---

## 4. Test 3: Fresh Knowledge-Base Indexing

### 4.1 Indexing Execution:
- **Command:**  
  `python scripts/index_kb.py --strategy section_aware --mock`
- **Runtime Target:** Clean runtime directory (`data/runtime/chroma_db`).
- **Exit Code:** `0` (Success).

```
==================================================
 CloudServe AI Support - Knowledge Base Indexer   
==================================================
KB Path:           data\kb\documentation.json
Strategy:          section_aware
Persist Directory: data\runtime\chroma_db
Collection Name:   cloudserve_kb
Embedding Mode:    Mock (Offline)
--------------------------------------------------
Successfully loaded 29 documentation articles.
Generated 116 chunks using strategy 'section_aware'.

Chunks by Category:
  • account         : 8 chunks
  • api             : 12 chunks
  • authentication  : 16 chunks
  • billing         : 12 chunks
  • data            : 12 chunks
  • deployment      : 16 chunks
  • integration     : 8 chunks
  • onboarding      : 8 chunks
  • performance     : 12 chunks
  • security        : 12 chunks

Connecting to ChromaDB and generating embeddings...
SUCCESS: Indexed 116 chunks into 'cloudserve_kb'.
Total documents in collection: 116
==================================================
```

### 4.2 Idempotency Verification:
Re-running the indexing command against the populated database confirmed proper idempotency, updating existing records without duplication or corruption (total remained exactly 116).

---

## 5. Test 4: SQLite Database Initialization & Audit Persistence

### 5.1 Verification Protocol:
- Initialized SQLite engine on a fresh, non-existent database file: `data/runtime/test_fresh.db`.
- Verified `_ensure_sqlite_parent_dir()` auto-created missing directories.
- Executed `init_db(engine)` to create the schema and verify table migrations.
- Tested complete record insertion and read-back using `DecisionAuditRepository` and `build_audit_entry()`.

### 5.2 Verification Result:
- Schema initialized with WAL journal mode and normal synchronization pragmas.
- Decision audit entry committed and read back with exact match across `ticket_id`, `original_route`, `predicted_intent`, and `processing_latency_ms`.
- **Status:** **PASS (100% self-initializing)**.

---

## 6. Test 5: Application Startup & Credential Safety

### 6.1 Application Startup Test:
- Initialized `TicketProcessor` using `create_ticket_processor()`.
- Processed a sample inbound ticket (`TEST-0001`).
- Verified:
  - System safely validated configuration.
  - Successfully scanned input safety and evaluated deterministic routing.
  - Correctly suppressed customer response on escalation (`final_customer_response is None`).
  - Successfully committed audit event to SQLite.
- **Handling of Missing API Key:** Verified that unconfigured `GROQ_API_KEY` raises a clear, descriptive `ConfigurationError` rather than leaking internal tracebacks or exposing secret placeholders.

---

## 7. Test 6: Automated Test Suite Execution

### 7.1 Test Execution in Clean-Room Environment:
- **Command:**  
  `pytest tests/unit --cov=src/cloudserve --cov-report=term`
- **Execution Date/Time:** November 22, 2026, 14:39:58 IST
- **Exit Code:** `0` (Success)
- **Execution Duration:** 68.13s (0:01:08)
- **Total Tests Run:** **452 passed, 0 failed, 0 skipped**.
- **Statement Code Coverage:** **92% across all 4,492 statements** (exceeding the 80% SLA).

### 7.2 Coverage Highlights:
- `src/cloudserve/routing/`: **100% Statement Coverage** (Zero untested policy branches).
- `src/cloudserve/guardrails/`: **98% Statement Coverage**.
- `src/cloudserve/classification/models.py`: **100% Statement Coverage**.
- `src/cloudserve/service.py`: **94% Statement Coverage**.

---

## 8. Test 7: Mock Smoke Evaluation

### 8.1 Unattended Evaluation Harness Execution:
- **Command:**  
  `python scripts/evaluate.py --input data/dev/manual_smoke_1.json --output-dir reports/clean_smoke_eval --mode mock`
- **Exit Code:** `0` (Success).
- **Total Tickets Processed:** 1 / 1.
- **Persisted Audit Records:** 1 (100% reconciliation).
- **Generated Artifacts:**
  - `run_metadata.json`
  - `audit_reconciliation.json`
  - `summary_metrics.json`
  - `per_ticket_results.jsonl`
  - `per_ticket_results.csv`
  - `human_review_packet.json`
  - `human_review_template.csv`
  - `human_review_metadata.json`
  - `EVALUATION_REPORT.md`

---

## 9. Test 8: Public Repository Data Review & Privacy Audit

An exhaustive regex and keyword scan was executed across all datasets proposed for public inclusion in GitHub:

| Dataset File | File Size | Real Customer PII | Live Credentials | Recommendation |
| :--- | :---: | :---: | :---: | :---: |
| `data/dev/development_tickets.json` | 453,003 B | 0 | 0 | **Safe to Publish** (Synthetic tickets) |
| `data/dev/manual_smoke_1.json` | 903 B | 0 | 0 | **Safe to Publish** |
| `data/dev/manual_smoke_5.json` | 4,720 B | 0 | 0 | **Safe to Publish** |
| `data/val/validation_tickets.json` | 71,999 B | 0 | 0 | **Safe to Publish** (Synthetic tickets) |
| `data/val/ground_truth_responses.json` | 221,693 B | 0 | 0 | **Safe to Publish** (Synthetic reference answers) |
| `data/kb/documentation.json` | 40,771 B | 0 | 0 | **Safe to Publish** (Synthetic documentation) |

*Finding:* All data is synthetic and non-confidential. No live API keys, personal email addresses, or phone numbers exist.

---

## 10. Disclosed Limitations

1. **Working Directory for CLI Scripts:** Running CLI tools (`scripts/evaluate.py`, `scripts/index_kb.py`) expects execution from the repository root unless explicitly specifying paths via environment variables.
2. **P95 Latency SLA Target:** As documented in the main technical dossier, live LLM evaluation runs exhibited P95 latency between 4.09s and 4.55s against the < 3.0s SLA target.
3. **Empty-Set Validation Quality Metrics:** `data/val/ground_truth_responses.json` maps reference responses exclusively to development IDs (`DEV-xxxx`), so automated semantic metrics for validation tickets evaluate 0 tickets.

---

## 11. Final Verification Conclusion

The CloudServe AI Customer Support System repository has been proven to be **FULLY REPRODUCIBLE** in an isolated clean-room environment. A new developer can clone the repository, create a virtual environment, install dependencies, index the knowledge base, run all 452 tests, and execute evaluation CLI harnesses without relying on any local artifacts or machine-specific paths.
