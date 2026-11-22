"""Targeted Live Security Regression Script for CloudServe A3.36.3.

Tests a curated set of 7 tickets covering security incidents, routine deprovisioning,
and authentication recovery.
"""

import json
import time
from pathlib import Path

from cloudserve.classification.classifier import TicketClassifier
from cloudserve.generation.llm_client import LangChainGroqClient
from cloudserve.ingestion.parsers import parse_evaluation_dataset
from cloudserve.service import create_ticket_processor


def run_regression() -> None:
    llm_client = LangChainGroqClient()
    classifier = TicketClassifier(llm_client=llm_client)
    processor = create_ticket_processor(classifier=classifier, generator_llm=llm_client)

    dataset_path = Path("data/dev/development_tickets.json")
    dataset_content = dataset_path.read_text(encoding="utf-8")
    dataset = parse_evaluation_dataset(dataset_content)
    ticket_map = {t.ticket_id: t for t in dataset.tickets}

    target_ids = [
        "DEV-0234",  # Former employee, unauthorized API activity post-deprovisioning
        "DEV-0015",  # Positive security-incident case (compromise post-termination)
        "DEV-0085",  # Positive security incident (unrecognized audit activity)
        "DEV-0019",  # Routine account-access / offboarding case
        "DEV-0094",  # Routine account-access / offboarding case
        "DEV-0045",  # Authentication failure / password lockout
        "DEV-0035",  # Suspected credential-compromise case
    ]

    results = []

    print("Initializing LangChainGroqClient and TicketProcessor...\n")

    for tid in target_ids:
        gt = ticket_map[tid]
        print(f"--- Running Live Inference for {tid} ---")
        print(f"Subject: {gt.subject}")
        print(f"Body: {gt.body}")

        t0 = time.perf_counter()
        res = processor.process(gt.to_domain_ticket())
        latency_ms = (time.perf_counter() - t0) * 1000

        db_rec = processor.audit_repository.get_by_audit_id(res.audit_id)
        db_persisted = db_rec is not None

        cls_intent = res.classification.intent.value if res.classification else None
        cls_intent_conf = res.classification.intent_confidence if res.classification else None
        cls_urg = res.classification.urgency.value if res.classification else None
        cls_urg_conf = res.classification.urgency_confidence if res.classification else None

        result_summary = {
            "ticket_id": tid,
            "is_synthetic": gt.is_synthetic,
            "expected_intent": gt.labels.expected_intent,
            "expected_urgency": gt.labels.expected_urgency,
            "expected_route": gt.labels.expected_route,
            "predicted_intent": cls_intent,
            "intent_confidence": cls_intent_conf,
            "predicted_urgency": cls_urg,
            "urgency_confidence": cls_urg_conf,
            "actual_route": res.routing_decision.route.value,
            "reason_code": res.routing_decision.reason_code.value,
            "target_specialist_group": (
                res.routing_decision.escalation_payload.target_specialist_group
                if res.routing_decision.escalation_payload
                else None
            ),
            "generation_bypassed": res.generated_response is None,
            "final_customer_response": (
                res.final_customer_response.text if res.final_customer_response else None
            ),
            "delivery_status": res.delivery_status.value,
            "latency_ms": round(latency_ms, 2),
            "audit_id": str(res.audit_id),
            "persistence_status": res.persistence_status.value,
            "db_persisted": db_persisted,
            "db_original_route": db_rec.original_route if db_rec else None,
            "db_effective_route": db_rec.effective_route if db_rec else None,
            "db_reason_code": db_rec.reason_code if db_rec else None,
            "rationale": res.classification.reasoning if res.classification else None,
        }
        results.append(result_summary)

        route_str = f"{result_summary['actual_route']} [{result_summary['reason_code']}]"
        print(f"Result: {cls_intent} ({cls_urg}) -> {route_str}")
        print(
            f"Generation bypassed: {result_summary['generation_bypassed']} | "
            f"Delivery: {result_summary['delivery_status']} | "
            f"DB: {result_summary['db_persisted']}\n"
        )

    out_file = Path("reports/targeted_security_regression_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"All 7 live test tickets completed! Results saved to {out_file}")


if __name__ == "__main__":
    run_regression()
