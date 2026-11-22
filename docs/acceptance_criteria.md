# CloudServe AI Support System - Official Acceptance Criteria (A1 - A12)

> **Source:** `docs/reference/Build_Specification.docx` (Section 02: "The twelve acceptance criteria")  
> **Status:** All criteria are Pass/Fail. No partial credit. No criterion may be marked as passed until verified through automated tests.

---

## The Twelve Acceptance Criteria

| ID | Criterion | How it is Judged |
| :--- | :--- | :--- |
| **A1** | The system runs from a clean checkout by following your README, using the commands you documented. | Cloned into an empty directory, instructions followed literally. It either starts or it does not. |
| **A2** | Tickets from all four channels are ingested and normalised into one internal representation. | A ticket from each channel is passed in. All four are handled without channel-specific breakage. |
| **A3** | Every ticket is classified for intent and urgency, with a numeric confidence attached. | The output carries a class and a confidence between zero and one. (Preserves all 22 intent classes and 3 urgency levels from the schema). |
| **A4** | Retrieval runs against the supplied documentation corpus and returns identifiable source passages. | The returned sources resolve back to real passages in the corpus, not to invented references. (29 KB articles). |
| **A5** | Routing applies a threshold you set, and the same input produces the same routing decision. | Run the same ticket twice. The decision does not change. |
| **A6** | Generated answers carry citations that resolve to the passages actually retrieved. | Citations are followed and checked against the text they claim to support. |
| **A7** | At least one guardrail can block a response, and does so when triggered. | A ticket engineered to trigger it is submitted. The response is blocked rather than sent. |
| **A8** | Every automated decision is written to a persistent log with the required fields. | The log is opened. Decisions are counted against tickets processed and must reconcile. |
| **A9** | The system processes the full evaluation set in a single unattended run. | Started once and left alone. No manual intervention, no restarts, no skipped tickets. Then re-run against a file you have never seen (accepts arbitrary `--input` and `--output` paths). |
| **A10** | That run produces a metrics report without further manual work. | A file appears containing the measures defined in the evaluation framework. |
| **A11** | The system handles failure without crashing: no retrieval hit, provider timeout or outage, rate limiting, malformed input. | Each condition is induced, including disconnecting the model provider entirely. The system degrades and continues rather than stopping. |
| **A12** | Tests run with a single documented command and pass. | The command in your README is run. The suite executes and reports (`pytest`). |

---

## Component-Level Behavioral Specifications

*(From `Build_Specification.docx` Section 03)*

### 1. Ingest
* Accepts tickets from email, live chat, documentation comments and the community forum.
* Produces one normalised internal representation regardless of source channel.
* Preserves the original text and the channel, because both matter downstream.
* Handles missing fields, unusual characters and empty bodies without failing.

### 2. Classify
* Assigns an intent category (all 22 intent classes preserved) and an urgency level (`low`, `medium`, `high`) to every ticket.
* Attaches a numeric confidence that reflects the actual probability of being correct.
* Returns a defined fallback rather than raising an exception when it cannot classify.
* Records the alternatives it considered, not only the option it chose.

### 3. Retrieve
* Searches the supplied documentation corpus (29 articles in `data/kb/documentation.json`) and returns ranked passages with scores.
* Returns identifiers that resolve to the real corpus, so citations can be verified.
* Applies a relevance threshold and returns nothing rather than something irrelevant.
* Chunking strategy is a design decision that must be justified and explained.

### 4. Route
* Decides between answering automatically (`auto_respond`) and escalating (`escalate`), using a threshold determined from data.
* Is deterministic: the same input yields the same decision.
* Records the reason for the decision in language a support manager could read.
* Escalations carry the drafted summary and the retrieved sources with them.

### 5. Generate
* Produces an answer grounded in the retrieved passages, with citations attached to claims.
* States plainly when it does not know rather than filling the gap.
* Separates the ticket content from your instructions so that customer text cannot redirect the system.
* Produces output in a defined structure your code can parse reliably.

### 6. Validate (Guardrails)
* Runs on every generated response before it is released, not only in testing.
* Checks at minimum for private data (PII) and for claims unsupported by the retrieved sources.
* Can block. A guardrail that only warns is not a guardrail.
* Records what it checked and what it found, whether or not it blocked.
