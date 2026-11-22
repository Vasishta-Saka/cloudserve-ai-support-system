# Classification Prompts Register

This directory contains version-controlled prompt templates for ticket intent and urgency classification.

## Version History

### `PR-CLS-v1.0` (File: `v1.txt`)
* **Date:** 2026-11-18
* **Target Task:** Intent (22 categories) & Urgency (3 tiers) classification with candidate alternatives.
* **Author:** Forward Deployed Engineering Team
* **Description:** System prompt defining the complete 22-intent taxonomy and 3 urgency levels. Employs few-shot calibration examples drawn strictly from `data/dev/development_tickets.json`. Enforces structured JSON output separating `intent_confidence` and `urgency_confidence` alongside an auditable `rationale`.
