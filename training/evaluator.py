"""
Hypothesis evaluation logic.

Evaluates structured hypotheses against actual results,
computing correctness and specificity scores.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from enum import Enum


class ClaimType(str, Enum):
    """Types of claims that can be evaluated."""
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    CLUSTER = "cluster"
    ANOMALY = "anomaly"
    COMPARISON = "comparison"
    EXISTENCE = "existence"
    COUNT = "count"
    TREND = "trend"
    GENERAL = "general"


@dataclass
class StructuredHypothesis:
    """
    A structured, testable hypothesis.
    
    The model is prompted to output hypotheses in this format
    to enable programmatic evaluation.
    """
    text: str  # Human-readable hypothesis text
    claim_type: ClaimType
    
    # Claim-specific fields (not all apply to every type)
    column_a: Optional[str] = None
    column_b: Optional[str] = None
    direction: Optional[Literal["positive", "negative", "none"]] = None
    magnitude_range: Optional[tuple[float, float]] = None
    value_range: Optional[tuple[float, float]] = None
    expected_count: Optional[int] = None
    count_range: Optional[tuple[int, int]] = None
    cluster_id: Optional[int] = None
    threshold: Optional[float] = None
    reasoning: Optional[str] = None
    
    # Evaluation results (filled after evaluation)
    is_correct: Optional[bool] = None
    actual_value: Optional[Any] = None
    specificity_score: int = 0
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "claim_type": self.claim_type.value,
            "column_a": self.column_a,
            "column_b": self.column_b,
            "direction": self.direction,
            "magnitude_range": list(self.magnitude_range) if self.magnitude_range else None,
            "value_range": list(self.value_range) if self.value_range else None,
            "expected_count": self.expected_count,
            "count_range": list(self.count_range) if self.count_range else None,
            "cluster_id": self.cluster_id,
            "threshold": self.threshold,
            "reasoning": self.reasoning,
            "is_correct": self.is_correct,
            "actual_value": self.actual_value,
            "specificity_score": self.specificity_score,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StructuredHypothesis":
        claim_type = ClaimType(data.get("claim_type", "general"))
        return cls(
            text=data.get("text", ""),
            claim_type=claim_type,
            column_a=data.get("column_a"),
            column_b=data.get("column_b"),
            direction=data.get("direction"),
            magnitude_range=tuple(data["magnitude_range"]) if data.get("magnitude_range") else None,
            value_range=tuple(data["value_range"]) if data.get("value_range") else None,
            expected_count=data.get("expected_count"),
            count_range=tuple(data["count_range"]) if data.get("count_range") else None,
            cluster_id=data.get("cluster_id"),
            threshold=data.get("threshold"),
            reasoning=data.get("reasoning"),
        )


class HypothesisEvaluator:
    """
    Evaluates structured hypotheses against actual results.
    
    Computes:
    - Correctness: Does the hypothesis match the actual result?
    - Specificity score: How specific/testable was the hypothesis?
    """
    
    def __init__(self, tolerance: float = 0.1):
        """
        Args:
            tolerance: Relative tolerance for numeric comparisons (default 10%)
        """
        self.tolerance = tolerance
    
    def evaluate(
        self,
        hypothesis: StructuredHypothesis,
        actual_results: dict,
        source_data: Optional[Any] = None,
    ) -> StructuredHypothesis:
        """
        Evaluate a single hypothesis against actual results.
        
        Args:
            hypothesis: The structured hypothesis to evaluate
            actual_results: The actual output from tool execution
            source_data: Optional DataFrame for additional checks
            
        Returns:
            The hypothesis with is_correct, actual_value, and specificity_score filled
        """
        # Compute specificity score first
        hypothesis.specificity_score = self._compute_specificity(hypothesis)
        
        # Evaluate based on claim type
        evaluator = getattr(self, f"_evaluate_{hypothesis.claim_type.value}", None)
        if evaluator:
            hypothesis = evaluator(hypothesis, actual_results, source_data)
        else:
            # Fallback: mark as needing manual review
            hypothesis.is_correct = None
        
        return hypothesis
    
    def evaluate_batch(
        self,
        hypotheses: list[StructuredHypothesis],
        actual_results: dict,
        source_data: Optional[Any] = None,
    ) -> list[StructuredHypothesis]:
        """Evaluate multiple hypotheses."""
        return [self.evaluate(h, actual_results, source_data) for h in hypotheses]
    
    def _compute_specificity(self, h: StructuredHypothesis) -> int:
        """
        Compute specificity score (0-5).
        
        Higher scores for more specific, testable claims.
        """
        score = 0
        
        # +1 for naming specific columns
        if h.column_a:
            score += 1
        if h.column_b:
            score += 1
        
        # +1 for providing numeric range
        if h.magnitude_range or h.value_range or h.count_range:
            score += 1
            # +1 bonus for narrow range
            if h.magnitude_range:
                width = abs(h.magnitude_range[1] - h.magnitude_range[0])
                if width < 0.3:  # Narrow range
                    score += 1
            if h.value_range:
                if h.value_range[0] != 0:  # Not starting from zero
                    width = abs(h.value_range[1] - h.value_range[0]) / abs(h.value_range[0])
                    if width < 0.5:  # Less than 50% relative width
                        score += 1
        
        # +1 for providing reasoning/mechanism
        if h.reasoning and len(h.reasoning) > 20:
            score += 1
        
        return min(score, 5)  # Cap at 5
    
    def _evaluate_correlation(
        self,
        h: StructuredHypothesis,
        results: dict,
        source_data: Optional[Any],
    ) -> StructuredHypothesis:
        """Evaluate correlation claims."""
        # Try to find correlation in results
        actual_corr = self._extract_correlation(results, h.column_a, h.column_b)
        
        if actual_corr is None and source_data is not None:
            # Compute from source data
            try:
                if h.column_a in source_data.columns and h.column_b in source_data.columns:
                    actual_corr = float(source_data[h.column_a].corr(source_data[h.column_b]))
            except Exception:
                pass
        
        if actual_corr is None:
            h.is_correct = None
            return h
        
        h.actual_value = round(actual_corr, 4)
        
        # Check direction
        direction_correct = True
        if h.direction:
            if h.direction == "positive" and actual_corr <= 0:
                direction_correct = False
            elif h.direction == "negative" and actual_corr >= 0:
                direction_correct = False
            elif h.direction == "none" and abs(actual_corr) > 0.1:
                direction_correct = False
        
        # Check magnitude range
        magnitude_correct = True
        if h.magnitude_range:
            low, high = h.magnitude_range
            if not (low <= actual_corr <= high):
                magnitude_correct = False
        
        h.is_correct = direction_correct and magnitude_correct
        return h
    
    def _evaluate_distribution(
        self,
        h: StructuredHypothesis,
        results: dict,
        source_data: Optional[Any],
    ) -> StructuredHypothesis:
        """Evaluate distribution claims (mean, median, percentiles)."""
        if not h.column_a:
            h.is_correct = None
            return h
        
        actual_value = None
        
        # Try to extract from results
        if "describe" in results:
            desc = results["describe"]
            if h.column_a in desc:
                actual_value = desc[h.column_a].get("mean")
        
        # Fallback to source data
        if actual_value is None and source_data is not None:
            try:
                if h.column_a in source_data.columns:
                    actual_value = float(source_data[h.column_a].mean())
            except Exception:
                pass
        
        if actual_value is None:
            h.is_correct = None
            return h
        
        h.actual_value = round(actual_value, 4)
        
        # Check if actual falls in expected range
        if h.value_range:
            low, high = h.value_range
            h.is_correct = low <= actual_value <= high
        else:
            h.is_correct = None  # No range to check against
        
        return h
    
    def _evaluate_cluster(
        self,
        h: StructuredHypothesis,
        results: dict,
        source_data: Optional[Any],
    ) -> StructuredHypothesis:
        """Evaluate clustering claims."""
        clusters = results.get("clusters", {})
        
        if not clusters:
            h.is_correct = None
            return h
        
        # Check count claims
        if h.count_range:
            actual_count = len(clusters)
            h.actual_value = actual_count
            low, high = h.count_range
            h.is_correct = low <= actual_count <= high
        elif h.expected_count is not None:
            actual_count = len(clusters)
            h.actual_value = actual_count
            h.is_correct = actual_count == h.expected_count
        elif h.cluster_id is not None:
            # Check specific cluster size
            cluster_key = str(h.cluster_id)
            if cluster_key in clusters:
                h.actual_value = clusters[cluster_key]
                if h.value_range:
                    low, high = h.value_range
                    h.is_correct = low <= clusters[cluster_key] <= high
                else:
                    h.is_correct = None
            else:
                h.is_correct = False
        else:
            h.is_correct = None
        
        return h
    
    def _evaluate_anomaly(
        self,
        h: StructuredHypothesis,
        results: dict,
        source_data: Optional[Any],
    ) -> StructuredHypothesis:
        """Evaluate anomaly detection claims."""
        # Look for anomaly count in results
        anomaly_count = None
        
        if "anomaly_count" in results:
            anomaly_count = results["anomaly_count"]
        elif "anomalies" in results:
            anomaly_count = len(results["anomalies"])
        
        if anomaly_count is None:
            h.is_correct = None
            return h
        
        h.actual_value = anomaly_count
        
        if h.count_range:
            low, high = h.count_range
            h.is_correct = low <= anomaly_count <= high
        elif h.expected_count is not None:
            # Allow some tolerance for exact count
            h.is_correct = abs(anomaly_count - h.expected_count) <= max(1, h.expected_count * 0.1)
        else:
            h.is_correct = None
        
        return h
    
    def _evaluate_comparison(
        self,
        h: StructuredHypothesis,
        results: dict,
        source_data: Optional[Any],
    ) -> StructuredHypothesis:
        """Evaluate comparison claims (A > B, A is largest, etc.)."""
        if not h.column_a:
            h.is_correct = None
            return h
        
        # This requires context-specific logic
        # For now, mark as needing review
        h.is_correct = None
        return h
    
    def _evaluate_existence(
        self,
        h: StructuredHypothesis,
        results: dict,
        source_data: Optional[Any],
    ) -> StructuredHypothesis:
        """Evaluate existence claims (there are outliers, there is a pattern, etc.)."""
        # Check for boolean indicators in results
        if h.threshold is not None and h.column_a and source_data is not None:
            try:
                col = source_data[h.column_a]
                exists = (col > h.threshold).any()
                h.actual_value = int((col > h.threshold).sum())
                h.is_correct = exists
            except Exception:
                h.is_correct = None
        else:
            h.is_correct = None
        
        return h
    
    def _evaluate_count(
        self,
        h: StructuredHypothesis,
        results: dict,
        source_data: Optional[Any],
    ) -> StructuredHypothesis:
        """Evaluate count claims."""
        actual_count = None
        
        # Try various result keys
        for key in ["count", "total", "n", "matching_rows", "total_rows"]:
            if key in results:
                actual_count = results[key]
                break
        
        if actual_count is None:
            h.is_correct = None
            return h
        
        h.actual_value = actual_count
        
        if h.count_range:
            low, high = h.count_range
            h.is_correct = low <= actual_count <= high
        elif h.expected_count is not None:
            h.is_correct = actual_count == h.expected_count
        else:
            h.is_correct = None
        
        return h
    
    def _evaluate_trend(
        self,
        h: StructuredHypothesis,
        results: dict,
        source_data: Optional[Any],
    ) -> StructuredHypothesis:
        """Evaluate trend claims (increasing, decreasing, etc.)."""
        # Would need time series or ordered data
        h.is_correct = None
        return h
    
    def _evaluate_general(
        self,
        h: StructuredHypothesis,
        results: dict,
        source_data: Optional[Any],
    ) -> StructuredHypothesis:
        """Fallback for general claims - needs LLM-as-judge."""
        h.is_correct = None
        return h
    
    def _extract_correlation(
        self,
        results: dict,
        col_a: Optional[str],
        col_b: Optional[str],
    ) -> Optional[float]:
        """Extract correlation value from results dict."""
        # Check for correlation pairs
        if "pairs" in results:
            for pair in results["pairs"]:
                if (pair.get("col_a") == col_a and pair.get("col_b") == col_b) or \
                   (pair.get("col_a") == col_b and pair.get("col_b") == col_a):
                    return pair.get("r")
        
        # Check for correlation matrix
        if "correlations" in results and col_a and col_b:
            corr_matrix = results["correlations"]
            if col_a in corr_matrix and col_b in corr_matrix[col_a]:
                return corr_matrix[col_a][col_b]
        
        return None


def parse_hypothesis_response(response_text: str) -> list[StructuredHypothesis]:
    """
    Parse LLM response into structured hypotheses.
    
    Expects JSON array or numbered list format.
    """
    import json
    
    hypotheses = []
    
    # Try JSON parse first
    try:
        # Look for JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            data = json.loads(json_match.group())
            for item in data:
                if isinstance(item, dict):
                    hypotheses.append(StructuredHypothesis.from_dict(item))
            if hypotheses:
                return hypotheses
    except json.JSONDecodeError:
        pass
    
    # Fallback: parse numbered list
    lines = response_text.strip().split('\n')
    for line in lines:
        # Match "1. ...", "1) ...", "- ...", etc.
        match = re.match(r'^[\d\-\*\•]+[\.\)]\s*(.+)$', line.strip())
        if match:
            text = match.group(1).strip()
            if text:
                # Create a general hypothesis from text
                hypotheses.append(StructuredHypothesis(
                    text=text,
                    claim_type=ClaimType.GENERAL,
                ))
    
    return hypotheses
