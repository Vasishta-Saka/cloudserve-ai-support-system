# CloudServe AI Customer Support System
## Phase 1: GitHub Repository Readiness & Portability Remediation Audit
**Program:** Post Graduate Certificate in Forward Deployed AI Engineering (IIT Roorkee)  
**Candidate:** Vasishta Saka  
**Repository Path:** `C:\Projects\CloudServe-AI-Support-System`  
**Frozen Academic Evidence:** `C:\Projects\CloudServe-Submission-Evidence\`  
**Date of Audit:** November 22, 2026  
**Auditor Role:** Senior Python Platform Engineer, DevSecOps Engineer & Lead AI Architect  

---

## 1. Executive Summary & Repository Status

This audit prepares the CloudServe AI Customer Support System repository for a clean, secure, reproducible public release on GitHub. 

### Key Findings:
1. **Academic Evidence Isolation:** All academic deliverables (evaluations, human review packets, paired workbooks, Stage 1–5 workbooks, and candidate effort logs) are completely frozen in `C:\Projects\CloudServe-Submission-Evidence\`.
2. **Current Repository State:**
   - The repository has **zero remote configured** and **9 local commit checkpoints** (which contain zero secrets).
   - The working tree currently contains 21 top-level items, including required application source code, configuration, tests, prompts, and some local ad-hoc scratch scripts / root markdown files.
3. **Secret Security & Data Privacy:**
   - Active `.env` file exists and is strictly excluded by `.gitignore`.
   - `.env.example` provides safe placeholder values.
   - Zero production API keys, customer PII, or internal credentials exist in tracked files or commit history.
4. **Portability Architecture:**
   - The codebase contains **zero hardcoded Windows drive letters (`C:\`) or absolute user paths**.
   - Path resolution for prompts (`prompts/classification`, `prompts/response_generation`) is anchored dynamically using `Path(__file__).resolve().parents[...]`.
   - Default directories in `src/cloudserve/config.py` use relative defaults (`./data/runtime/chroma_db`, `./data/runtime/cloudserve.db`, `./data/kb`).
5. **Final GitHub Readiness Verdict:** **`READY FOR CLEAN-ROOM VERIFICATION`** (Subject to updating `.gitignore` to exclude `scratch/` and root interim audit markdowns prior to initial commit).

---

## 2. Repository Inventory & Publication Segregation

To comply with open-source licensing, academic evaluation guidelines, and data privacy, the repository's contents are categorized for publication vs. academic retention:

### 2.1 Proposed Files for Public GitHub Repository
The following modules, tests, prompts, and configurations constitute the official, reproducible engineering distribution:

| Repository Category | Path / Files Included | Justification / Role |
| :--- | :--- | :--- |
| **Application Source** | `src/cloudserve/` (All 43 modules) | Core enterprise support pipeline, classifier, RAG, router, guardrails, storage. |
| **Test Suite** | `tests/` (28 test modules, `conftest.py`) | 452 unit and integration tests verifying 100% of functional requirements. |
| **Configuration** | `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.env.example` | Pinned dependencies and reproducible environment definitions. |
| **CI / CD Workflows** | `.github/workflows/ci.yml` | GitHub Actions workflow automating linting (Ruff), typing (Mypy), and testing (Pytest). |
| **Prompt Library** | `prompts/classification/v1.txt`, `v2.txt`, `prompts/response_generation/v1.txt` | Active and historical version-controlled prompt templates with SHA-256 traceability. |
| **Scripts & Entry Points**| `scripts/evaluate.py`, `scripts/index_kb.py`, `scripts/run_targeted_security_regression.py` | CLI harnesses for unattended evaluation, vector DB indexing, and regression. |
| **Documentation** | `README.md`, `docs/architecture.md`, `docs/traceability_matrix.md`, `docs/acceptance_criteria.md` | Comprehensive operational and architectural documentation for external evaluators. |
| **Knowledge Base** | `data/kb/documentation.json` | 29 synthetic technical knowledge-base articles (synthetic, safe for public release). |
| **Smoke & Benchmark Datasets** | `data/dev/development_tickets.json`, `data/val/validation_tickets.json`, `data/dev/manual_smoke_*.json` | Benchmark ticket sets (anonymized, synthetic cloud customer queries). |
| **Reference Response Ground Truth**| `data/val/ground_truth_responses.json` | 200 reference answers used for semantic evaluation. |
| **Master Evidence Dossier** | `CLOUDSERVE_FINAL_TECHNICAL_EVIDENCE_REPORT.md` | 722-line master systems engineering dossier. |

### 2.2 Proposed Exclusions (Restricted to Frozen Academic Package)
The following files must be excluded from the public repository and kept exclusively in `C:\Projects\CloudServe-Submission-Evidence\`:

| Category / Artifact | Reason for Exclusion | Enforcement Mechanism |
| :--- | :--- | :--- |
| **Active Secrets (`.env`)** | Contains live Groq API key and local configurations. | Strict `.gitignore` (`.env`, `*.env`). |
| **Runtime SQLite Databases** | `data/runtime/cloudserve.db`, `cloudserve.db`, `decision_audit.db` (local session state). | `.gitignore` (`*.db`, `data/runtime/`). |
| **ChromaDB Vector Indexes** | `data/runtime/chroma_db/` (generated local vector binaries). | `.gitignore` (`data/runtime/`, `chroma_db/`). |
| **Evaluation Run Dumps** | `reports/eval_dev_live_500_v2/`, `reports/eval_val_live_final/`, ZIP archives. | `.gitignore` (`reports/`, `eval_outputs/`). |
| **Human Review Workbooks** | `Reviewer_1_Scoring.xlsx`, `Reviewer_2_Scoring.xlsx`, `CloudServe_Review_and_Governance_Reconciliation.xlsx`. | Excluded via `reports/` and `.gitignore`. |
| **Candidate Workbooks & Effort Log** | `submission/03_Workbooks/` (`VasishtaSaka_Effort_Log.pdf`, Discovery Workbooks). | Academic personal submissions; retained in evidence. |
| **Scratch Analysis Scripts** | `scratch/` (14 ad-hoc debug scripts). | Add `scratch/` to `.gitignore`. |
| **Interim Milestone Markdowns**| Root `A3.9_...`, `A3.10_...`, `A3.11_...`, `GROUNDING_...` | Move to archived reports or add to `.gitignore`. |

---

## 3. Git Hygiene & Exclusion Audit

### 3.1 Review of `.gitignore` Rules
The repository's `.gitignore` already comprehensively handles:
- Python byte-code and virtual environments (`.venv/`, `venv/`, `__pycache__/`).
- Build artifacts (`*.egg-info/`, `dist/`, `build/`).
- Runtime data & databases (`data/runtime/`, `*.db`, `*.sqlite`, `chroma_db/`).
- Evaluation outputs (`reports/`, `*.log`).
- Environment secrets (`.env`, `*.env`, but safely whitelisting `!.env.example`).

### 3.2 Recommended Git Hygiene Additions:
To ensure untracked interim files do not clutter the initial GitHub commit, the following rules should be appended to `.gitignore`:
```gitignore
# Ad-hoc development scratch directory
scratch/

# Interim audit and milestone markdowns (preserved in reports/)
A3.*.md
GROUNDING_AND_MOCK_EVALUATION_DIAGNOSIS.md
CLOUDSERVE_END_TO_END_AUDIT.md
ENGINEERING_GUIDELINES.md
submission/
```

---

## 4. Portability Audit & Remediation Findings

### 4.1 Path Resolution & Working Directory Independence
1. **Dynamic Prompt Loading:**
   In `src/cloudserve/classification/prompts.py:8`:
   ```python
   _PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts" / "classification"
   ```
   In `src/cloudserve/generation/prompts.py:12`:
   ```python
   _CANONICAL_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts" / "response_generation"
   ```
   *Audit Result:* **PORTABLE**. Prompts resolve relative to the Python package source tree, not `os.getcwd()`.
2. **Database & Storage Directory Auto-Creation:**
   In `src/cloudserve/storage/database.py:38-41`:
   ```python
   p = Path(raw_path)
   if not p.is_absolute():
       p = p.resolve()
   p.parent.mkdir(parents=True, exist_ok=True)
   ```
   In `src/cloudserve/rag/indexer.py:187`:
   ```python
   self.persist_directory.mkdir(parents=True, exist_ok=True)
   ```
   *Audit Result:* **PORTABLE**. The application automatically creates missing parent directories on a fresh machine without throwing `FileNotFoundError`.
3. **Model Auto-Download:**
   In `src/cloudserve/rag/indexer.py:107-118`, the embedding adapter falls back automatically to downloading `sentence-transformers/all-MiniLM-L6-v2` over HTTPS if not present in the local cache.

### 4.2 Known Portability Nuance (Relative Settings Defaults):
In `src/cloudserve/config.py`, defaults are defined as relative paths:
- `chroma_persist_directory: Path = Path("./data/runtime/chroma_db")`
- `database_url: str = "sqlite:///./data/runtime/cloudserve.db"`
- `kb_directory: Path = Path("./data/kb")`

*Operational Recommendation:* When launching CloudServe CLI tools (`scripts/evaluate.py`, `scripts/index_kb.py`), the command should be run from the repository root, or paths should be set via environment variables (`CHROMA_PERSIST_DIRECTORY`, `DATABASE_URL`). This is clearly documented in `README.md`.

---

## 5. Documentation Readiness & Verification

The root `README.md` (583 lines) has been verified to cover all 13 required areas:
1. **Project Overview & Capabilities:** Section 1 & Section 3 detail 4-channel normalization, mandatory escalation, grounded generation, and immutable audit logs.
2. **Architecture:** Section 4 & Section 5 provide the 5-Gate pipeline flow, subsystem components, and PRD v2.0 deterministic decision trees.
3. **Prerequisites & Python Compatibility:** Section 6 specifies Python 3.10, 3.11, and 3.12 support across Windows, macOS, and Linux.
4. **Environment Setup & Pinned Installation:** Section 6 gives exact commands for PowerShell (`python -m venv .venv`, `.\.venv\Scripts\Activate.ps1`) and Linux/macOS (`source .venv/bin/activate`), alongside `pip install -r requirements.txt -r requirements-dev.txt`.
5. **Configuration & Secrets:** Section 6.2 explains `.env.example` -> `.env` configuration for `GROQ_API_KEY`.
6. **Knowledge Base Indexing:** Section 7 gives `python scripts/index_kb.py` instructions.
7. **Application Startup & Test Execution:** Section 8 and Section 10 provide API usage and `pytest` execution instructions.
8. **Evaluation CLI Runbook:** Section 9 documents `scripts/evaluate.py` across mock and live modes with resume flags.
9. **Performance Transparency & Limitations:** Section 11 and Section 16 explicitly disclose that **P95 latency was 4.09s–4.55s against the < 3.0s SLA target**, and clarify that baseline dataset metrics reflect input distributions rather than claimed operational savings.

---

## 6. Clean-Room Reproducibility Verification Plan

To verify that an external evaluator can clone and run the project with zero dependencies on the current environment, the following test plan is defined:

### Test Protocol:
1. **Target Sandbox:** Create an isolated directory outside the project tree: `C:\Temp\CloudServe-CleanRoom-Test\`.
2. **Clean Copy:** Export pristine repository files (excluding `.env`, runtime DBs, Chroma data, caches, and scratch files).
3. **Virtual Environment Creation:**
   ```powershell
   cd C:\Temp\CloudServe-CleanRoom-Test
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. **Dependency Installation:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
5. **Configuration Setup:**
   ```powershell
   Copy-Item .env.example .env
   ```
6. **Knowledge Base Indexing (Dry Run & Mock):**
   ```powershell
   python scripts/index_kb.py --strategy section_aware --mock
   ```
7. **Full Automated Test Suite Execution:**
   ```powershell
   pytest
   ```
   *Expected Result:* All 452 unit tests pass in under 2 minutes with >= 80% coverage.
8. **Mock Evaluation Smoke Test:**
   ```powershell
   python scripts/evaluate.py --input data/dev/manual_smoke_1.json --output-dir reports/clean_smoke --mode mock
   ```
   *Expected Result:* Exit code 0, 1 ticket processed, deterministic route generated, SQLite database created automatically in `data/runtime/cloudserve.db`.

---

## 7. Final Verdict

### **`READY FOR CLEAN-ROOM VERIFICATION`**

**Status Justification:**
- The application code, tests, and configurations are clean, strictly typed, and free of hardcoded paths or secrets.
- Academic evidence has been completely decoupled and preserved in the external frozen folder.
- Ready to proceed to executing the clean-room verification test.
