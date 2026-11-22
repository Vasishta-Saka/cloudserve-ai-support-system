"""Evaluation metrics calculation module.

Implements calibration metrics (binned reliability, Expected Calibration Error - ECE)
and classification evaluation according to the Evaluation Framework and Governance standards.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from cloudserve.classification.models import IntentCategory, UrgencyLevel


class CalibrationBin(BaseModel):
    """Metrics for an individual confidence bin."""

    model_config = ConfigDict(frozen=True)

    bin_lower: float = Field(..., ge=0.0, le=1.0)
    bin_upper: float = Field(..., ge=0.0, le=1.0)
    sample_count: int = Field(..., ge=0)
    average_confidence: float = Field(..., ge=0.0, le=1.0)
    observed_accuracy: float = Field(..., ge=0.0, le=1.0)
    calibration_gap: float = Field(..., description="|avg_confidence - observed_accuracy|")
    is_statistically_reliable: bool = Field(
        ...,
        description="True if bin has sufficient sample size to evaluate governance target.",
    )


class CalibrationReport(BaseModel):
    """Comprehensive confidence calibration report."""

    model_config = ConfigDict(frozen=True)

    bins: list[CalibrationBin] = Field(default_factory=list)
    expected_calibration_error: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted Expected Calibration Error (ECE) across all bins.",
    )
    max_calibration_error: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum calibration gap across populated bins.",
    )
    total_samples: int = Field(..., ge=0)
    reliable_bins_count: int = Field(..., ge=0)
    meets_governance_target: bool = Field(
        ...,
        description="True if reliable bins meet target and no populated bins are sparse.",
    )
    governance_target_status: str = Field(
        default="inconclusive_due_to_small_bins",
        description="Status: meets_target, fails_target, or inconclusive_due_to_small_bins.",
    )
    reliable_bins_ece: float | None = Field(
        default=None,
        description="Weighted ECE restricted strictly to bins with >= min_samples_per_bin.",
    )
    reliable_bins_sample_denominator: int = Field(
        default=0,
        ge=0,
        description="Total samples contained in statistically reliable bins.",
    )
    reliable_bins_sample_coverage_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of total evaluation samples covered by reliable bins.",
    )
    notes: str = Field(
        ...,
        description="Auditable summary on reliability and sample sufficiency.",
    )


def calculate_confidence_calibration(
    confidences: Sequence[float],
    accuracies: Sequence[bool],
    n_bins: int = 5,
    min_samples_per_bin: int = 10,
    is_mock: bool = False,
) -> CalibrationReport:
    """Calculate binned reliability and Expected Calibration Error (ECE).

    Approved Governance Rule:
    - If ANY populated confidence bin has fewer than min_samples_per_bin (default: 10),
      the governance conclusion is 'inconclusive_due_to_small_bins'.
    - If all populated bins have at least min_samples_per_bin, evaluate the
      5-percentage-point target (gap <= 0.05).
    - Empty bins do NOT trigger the sparse-bin rule.
    - If no populated bins exist (0 samples), status is 'inconclusive_due_to_small_bins'.
    - In mock mode (is_mock=True), status is 'not_assessed_mock_mode'.

    Args:
        confidences: Sequence of predicted confidence scores in [0.0, 1.0].
        accuracies: Sequence of boolean correctness indicators matching confidences.
        n_bins: Number of equal-width probability bins (default: 5).
        min_samples_per_bin: Minimum sample count required for a bin to be considered
                             statistically reliable for governance evaluation (default: 10).
        is_mock: Set True for mock mode diagnostic runs to avoid claiming assessed LLM calibration.

    Returns:
        CalibrationReport: Detailed calibration analysis.
    """
    if len(confidences) != len(accuracies):
        raise ValueError(
            f"Length mismatch: {len(confidences)} confidences vs {len(accuracies)} accuracies."
        )

    bin_width = 1.0 / n_bins
    total_samples = len(confidences)
    if total_samples == 0:
        empty_bins = [
            CalibrationBin(
                bin_lower=round(i * bin_width, 4),
                bin_upper=round((i + 1) * bin_width, 4),
                sample_count=0,
                average_confidence=round((i + 0.5) * bin_width, 4),
                observed_accuracy=0.0,
                calibration_gap=0.0,
                is_statistically_reliable=False,
            )
            for i in range(n_bins)
        ]
        return CalibrationReport(
            bins=empty_bins,
            expected_calibration_error=0.0,
            max_calibration_error=0.0,
            total_samples=0,
            reliable_bins_count=0,
            meets_governance_target=False,
            governance_target_status="inconclusive_due_to_small_bins",
            reliable_bins_ece=None,
            reliable_bins_sample_denominator=0,
            reliable_bins_sample_coverage_pct=0.0,
            notes="Governance evaluation inconclusive: 0 samples provided (no populated bins).",
        )

    bins: list[CalibrationBin] = []
    weighted_ece_sum = 0.0
    max_gap = 0.0
    reliable_count = 0
    has_sparse_populated_bin = False
    all_reliable_meet_target = True

    for i in range(n_bins):
        lower = round(i * bin_width, 4)
        upper = round((i + 1) * bin_width, 4)

        # Include upper boundary in last bin
        if i == n_bins - 1:
            indices = [idx for idx, c in enumerate(confidences) if lower <= c <= upper]
        else:
            indices = [idx for idx, c in enumerate(confidences) if lower <= c < upper]

        count = len(indices)
        if count == 0:
            bins.append(
                CalibrationBin(
                    bin_lower=lower,
                    bin_upper=upper,
                    sample_count=0,
                    average_confidence=round((lower + upper) / 2.0, 4),
                    observed_accuracy=0.0,
                    calibration_gap=0.0,
                    is_statistically_reliable=False,
                )
            )
            continue

        bin_confs = [confidences[idx] for idx in indices]
        bin_accs = [1.0 if accuracies[idx] else 0.0 for idx in indices]

        avg_conf = round(sum(bin_confs) / count, 4)
        obs_acc = round(sum(bin_accs) / count, 4)
        gap = round(abs(avg_conf - obs_acc), 4)

        weighted_ece_sum += (count / total_samples) * gap
        if gap > max_gap:
            max_gap = gap

        is_reliable = count >= min_samples_per_bin
        if is_reliable:
            reliable_count += 1
            if gap > 0.05:
                all_reliable_meet_target = False
        else:
            has_sparse_populated_bin = True

        bins.append(
            CalibrationBin(
                bin_lower=lower,
                bin_upper=upper,
                sample_count=count,
                average_confidence=avg_conf,
                observed_accuracy=obs_acc,
                calibration_gap=gap,
                is_statistically_reliable=is_reliable,
            )
        )

    ece = round(weighted_ece_sum, 4)

    # Calculate reliable-bin-only metrics
    reliable_bins = [b for b in bins if b.is_statistically_reliable]
    reliable_denominator = sum(b.sample_count for b in reliable_bins)
    coverage_pct = (
        round(100.0 * reliable_denominator / total_samples, 2) if total_samples > 0 else 0.0
    )
    reliable_ece = (
        round(
            sum(b.sample_count * b.calibration_gap for b in reliable_bins) / reliable_denominator, 4
        )
        if reliable_denominator > 0
        else None
    )

    # Determine governance compliance & inconclusive status
    if is_mock:
        governance_status = "not_assessed_mock_mode"
        meets_target = False
        notes = (
            f"Mock confidence diagnostic (Full ECE: {ece:.3f}). "
            "Actual LLM model-calibration performance is not assessed in mock mode."
        )
    elif has_sparse_populated_bin or reliable_count == 0:
        governance_status = "inconclusive_due_to_small_bins"
        meets_target = False
        notes = (
            f"Governance evaluation inconclusive: at least one populated confidence bin "
            f"contains fewer than {min_samples_per_bin} observations (insufficient samples). "
            f"Cannot confirm whether the 5-percentage-point calibration target is met "
            f"(Full ECE: {ece:.3f}, Reliable Bins ECE: "
            f"{f'{reliable_ece:.3f}' if reliable_ece is not None else 'N/A'} "
            f"covering {coverage_pct:.1f}% of samples)."
        )
    elif all_reliable_meet_target:
        governance_status = "meets_target"
        meets_target = True
        notes = (
            f"Governance target met: all {reliable_count} populated bins exhibit "
            f"calibration gap <= 0.05 (Full ECE: {ece:.3f})."
        )
    else:
        governance_status = "fails_target"
        meets_target = False
        notes = (
            f"Governance target NOT met: at least one reliable bin exceeds 0.05 gap "
            f"(Full ECE: {ece:.3f}, Max gap: {max_gap:.3f})."
        )

    return CalibrationReport(
        bins=bins,
        expected_calibration_error=ece,
        max_calibration_error=round(max_gap, 4),
        total_samples=total_samples,
        reliable_bins_count=reliable_count,
        meets_governance_target=meets_target,
        governance_target_status=governance_status,
        reliable_bins_ece=reliable_ece,
        reliable_bins_sample_denominator=reliable_denominator,
        reliable_bins_sample_coverage_pct=coverage_pct,
        notes=notes,
    )


class ClassMetric(BaseModel):
    """Metrics for an individual classification category."""

    model_config = ConfigDict(frozen=True)

    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    support: int = Field(..., ge=0)


class ClassificationMetricsReport(BaseModel):
    """Comprehensive multi-class classification evaluation report."""

    model_config = ConfigDict(frozen=True)

    accuracy: float = Field(..., ge=0.0, le=1.0)
    macro_f1_represented: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Macro-F1 computed strictly over classes with support > 0.",
    )
    macro_f1_fixed_22: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Macro-F1 across all 22 taxonomy classes (zero-support classes score 0.0).",
    )
    zero_support_classes: list[str] = Field(default_factory=list)
    per_class: dict[str, ClassMetric] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    total_samples: int = Field(..., ge=0)


class UrgencyMetricsReport(BaseModel):
    """Urgency classification evaluation report."""

    model_config = ConfigDict(frozen=True)

    accuracy: float = Field(..., ge=0.0, le=1.0)
    macro_f1: float = Field(..., ge=0.0, le=1.0)
    per_class: dict[str, ClassMetric] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    total_samples: int = Field(..., ge=0)


def calculate_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    taxonomy: Sequence[str] | None = None,
) -> ClassificationMetricsReport:
    """Calculate multi-class classification metrics across the 22-class taxonomy.

    Args:
        y_true: Sequence of ground-truth intent labels.
        y_pred: Sequence of predicted intent labels.
        taxonomy: Optional explicit list of class names. Defaults to the 22 IntentCategory values.

    Returns:
        ClassificationMetricsReport with accuracy, represented macro-F1, fixed-22 macro-F1,
        per-class metrics, and confusion matrix.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: {len(y_true)} true labels vs {len(y_pred)} predicted labels."
        )

    # Use canonical 22 classes by default
    canonical_taxonomy = [cat.value for cat in IntentCategory]
    all_classes = list(taxonomy) if taxonomy is not None else list(canonical_taxonomy)

    # Also include any unexpected classes present in data so metrics are complete
    for label in list(y_true) + list(y_pred):
        if label not in all_classes:
            all_classes.append(label)

    total_samples = len(y_true)
    if total_samples == 0:
        return ClassificationMetricsReport(
            accuracy=0.0,
            macro_f1_represented=0.0,
            macro_f1_fixed_22=0.0,
            zero_support_classes=all_classes,
            per_class={
                cls_name: ClassMetric(precision=0.0, recall=0.0, f1=0.0, support=0)
                for cls_name in all_classes
            },
            confusion_matrix={c1: {c2: 0 for c2 in all_classes} for c1 in all_classes},
            total_samples=0,
        )

    # Compute TP, FP, FN, Support per class
    tp: dict[str, int] = {c: 0 for c in all_classes}
    fp: dict[str, int] = {c: 0 for c in all_classes}
    fn: dict[str, int] = {c: 0 for c in all_classes}
    support: dict[str, int] = {c: 0 for c in all_classes}

    # Confusion matrix: cm[true][pred] = count
    confusion_matrix: dict[str, dict[str, int]] = {
        c1: {c2: 0 for c2 in all_classes} for c1 in all_classes
    }

    correct_total = 0
    for yt, yp in zip(y_true, y_pred, strict=True):
        if yt == yp:
            correct_total += 1
            if yt in tp:
                tp[yt] += 1
        else:
            if yp in fp:
                fp[yp] += 1
            if yt in fn:
                fn[yt] += 1
        if yt in support:
            support[yt] += 1
        if yt in confusion_matrix and yp in confusion_matrix[yt]:
            confusion_matrix[yt][yp] += 1

    accuracy = round(correct_total / total_samples, 4)

    per_class: dict[str, ClassMetric] = {}
    f1_represented: list[float] = []
    zero_support: list[str] = []

    for c in all_classes:
        s = support[c]
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / s if s > 0 else 0.0
        f = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

        p_round = round(p, 4)
        r_round = round(r, 4)
        f_round = round(f, 4)

        per_class[c] = ClassMetric(
            precision=p_round,
            recall=r_round,
            f1=f_round,
            support=s,
        )

        if s > 0:
            f1_represented.append(f_round)
        else:
            zero_support.append(c)

    macro_f1_represented = (
        round(sum(f1_represented) / len(f1_represented), 4) if f1_represented else 0.0
    )

    # Fixed 22 macro-F1: strictly over the 22 canonical taxonomy classes
    fixed_22_f1s = [per_class[c].f1 for c in canonical_taxonomy if c in per_class]
    macro_f1_fixed_22 = (
        round(sum(fixed_22_f1s) / len(canonical_taxonomy), 4) if canonical_taxonomy else 0.0
    )

    return ClassificationMetricsReport(
        accuracy=accuracy,
        macro_f1_represented=macro_f1_represented,
        macro_f1_fixed_22=macro_f1_fixed_22,
        zero_support_classes=zero_support,
        per_class=per_class,
        confusion_matrix=confusion_matrix,
        total_samples=total_samples,
    )


def calculate_urgency_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> UrgencyMetricsReport:
    """Calculate classification metrics for the 3 urgency tiers (low, medium, high)."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: {len(y_true)} true labels vs {len(y_pred)} predicted labels."
        )

    taxonomy = [level.value for level in UrgencyLevel]
    total_samples = len(y_true)

    if total_samples == 0:
        return UrgencyMetricsReport(
            accuracy=0.0,
            macro_f1=0.0,
            per_class={
                cls_name: ClassMetric(precision=0.0, recall=0.0, f1=0.0, support=0)
                for cls_name in taxonomy
            },
            confusion_matrix={c1: {c2: 0 for c2 in taxonomy} for c1 in taxonomy},
            total_samples=0,
        )

    tp: dict[str, int] = {c: 0 for c in taxonomy}
    fp: dict[str, int] = {c: 0 for c in taxonomy}
    fn: dict[str, int] = {c: 0 for c in taxonomy}
    support: dict[str, int] = {c: 0 for c in taxonomy}

    confusion_matrix: dict[str, dict[str, int]] = {
        c1: {c2: 0 for c2 in taxonomy} for c1 in taxonomy
    }

    correct_total = 0
    for yt, yp in zip(y_true, y_pred, strict=True):
        if yt == yp:
            correct_total += 1
            if yt in tp:
                tp[yt] += 1
        else:
            if yp in fp:
                fp[yp] += 1
            if yt in fn:
                fn[yt] += 1
        if yt in support:
            support[yt] += 1
        if yt in confusion_matrix and yp in confusion_matrix[yt]:
            confusion_matrix[yt][yp] += 1

    accuracy = round(correct_total / total_samples, 4)

    per_class: dict[str, ClassMetric] = {}
    f1_list: list[float] = []

    for c in taxonomy:
        s = support[c]
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / s if s > 0 else 0.0
        f = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

        p_round = round(p, 4)
        r_round = round(r, 4)
        f_round = round(f, 4)

        per_class[c] = ClassMetric(
            precision=p_round,
            recall=r_round,
            f1=f_round,
            support=s,
        )
        f1_list.append(f_round)

    macro_f1 = round(sum(f1_list) / len(f1_list), 4) if f1_list else 0.0

    return UrgencyMetricsReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=confusion_matrix,
        total_samples=total_samples,
    )
