from typing import Sequence, Any, Dict, List, Set, Tuple, Mapping
from collections import Counter
from scipy.stats import pearsonr, spearmanr
import math
"""
'Evaluating LLM Understanding via Structured Tabular Decision Simulations'
"""
class STaDSMetrics:
    @staticmethod
    def _validate_and_align(
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
        missing_token: str = "<MISSING>"
    ) -> List[Tuple[Any, Any]]:
        """
        Aligns predictions with ground truth handling length mismatches.
        - Truncates extra predictions.
        - Pads missing predictions with <MISSING>.
        """
        gold_iter = list(ground_truths)
        pred_iter = list(predictions)

        if len(pred_iter) > len(gold_iter):
            pred_iter = pred_iter[:len(gold_iter)]
        elif len(pred_iter) < len(gold_iter):
            missing = len(gold_iter) - len(pred_iter)
            pred_iter.extend([missing_token] * missing)

        return list(zip(pred_iter, gold_iter))

    @staticmethod
    def accuracy(predictions: Sequence[Any], ground_truths: Sequence[Any]) -> float:
        aligned = STaDSMetrics._validate_and_align(predictions, ground_truths)
        if not aligned:
            return 0.0
        correct = sum(1 for pred, gold in aligned if pred == gold)
        return float(correct / len(aligned))

    @staticmethod
    def f1_score(
        predictions: Sequence[Any], 
        ground_truths: Sequence[Any], 
        average: str = "macro"
    ) -> float:
        aligned = STaDSMetrics._validate_and_align(predictions, ground_truths)
        if not aligned:
            return 0.0

        gold_labels = set(g for _, g in aligned)
        # Track stats only for known gold labels
        label2stats = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in gold_labels}

        for pred, gold in aligned:
            if pred == gold:
                label2stats[gold]["tp"] += 1
            else:
                # It is a miss for the gold class
                label2stats[gold]["fn"] += 1
                
                # It is a false alarm for the predicted class ONLY if 
                # that predicted class is actually a valid gold label.
                # Hallucinated labels do not increase FP count for valid classes.
                if pred in label2stats:
                    label2stats[pred]["fp"] += 1

        # Compute per-label F1
        per_label_f1 = []
        total_tp, total_fp, total_fn = 0, 0, 0

        for lbl, s in label2stats.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            denom = (2 * tp + fp + fn)
            score = (2 * tp / denom) if denom > 0 else 0.0
            per_label_f1.append(score)

        if average == "macro":
            return sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
        elif average == "micro":
            denom = (2 * total_tp + total_fp + total_fn)
            return (2 * total_tp / denom) if denom > 0 else 0.0
        else:
            raise ValueError("average must be 'macro' or 'micro'")

    @staticmethod
    def length_metrics(predictions: Sequence[Any], ground_truths: Sequence[Any]) -> Dict[str, float]:
        pred_len = len(predictions)
        gold_len = len(ground_truths)
        
        if pred_len == 0:
            return {"coverage_ratio": 0.0, "length_f1": 0.0}

        # Precision: How many of my outputs were 'valid slots' (up to gold length)
        used = min(pred_len, gold_len)
        precision = used / pred_len
        recall = used / gold_len
        
        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
            
        return {
            "coverage_ratio": pred_len / gold_len if gold_len else 0.0,
            "length_f1": f1
        }

    @staticmethod
    def integrity_metrics(predictions: Sequence[Any], ground_truths: Sequence[Any]) -> Dict[str, float]:
        pred_set = set(predictions)
        gold_set = set(ground_truths)
        
        # Unknown Label Rate: % of preds that are NOT in the gold set at all
        if not predictions:
            unknown_rate = 0.0
        else:
            unknown = [p for p in predictions if p not in gold_set]
            unknown_rate = len(unknown) / len(predictions)

        # Class Presence F1 (Recall of the SET of classes found)
        intersection = len(pred_set & gold_set)
        label_set_coverage = intersection / len(gold_set) if gold_set else 1.0
        class_presence_precision = intersection / len(pred_set) if pred_set else 1.0
        
        cp_f1 = 0.0
        if (label_set_coverage + class_presence_precision) > 0:
            cp_f1 = 2 * label_set_coverage * class_presence_precision / (label_set_coverage + class_presence_precision)

        return {
            "unknown_label_rate": unknown_rate,
            "class_presence_f1": cp_f1,
            "set_jaccard": intersection / len(pred_set | gold_set) if (pred_set | gold_set) else 1.0
        }

    @staticmethod
    def penalised_accuracy(
        predictions: Sequence[Any], 
        ground_truths: Sequence[Any], 
        alpha: float = .5, 
        beta: float = .5
    ) -> float:
        """
        Paper Metric: Accuracy penalized by Length F1 and Unknown Label Rate.
        Score = max(0, Acc - alpha*(1 - LengthF1) - beta*(UnknownRate))
        """
        base_acc = STaDSMetrics.accuracy(predictions, ground_truths)
        l_metrics = STaDSMetrics.length_metrics(predictions, ground_truths)
        i_metrics = STaDSMetrics.integrity_metrics(predictions, ground_truths)
        
        penalty = (alpha * (1.0 - l_metrics["length_f1"])) + (beta * i_metrics["unknown_label_rate"])
        return max(0.0, base_acc - penalty)

    @staticmethod
    def compute_all(predictions: Sequence[Any], ground_truths: Sequence[Any]) -> Dict[str, float]:
        return {
            "accuracy": STaDSMetrics.accuracy(predictions, ground_truths),
            "f1_macro": STaDSMetrics.f1_score(predictions, ground_truths, "macro"),
            "penalised_accuracy": STaDSMetrics.penalised_accuracy(predictions, ground_truths),
            **STaDSMetrics.length_metrics(predictions, ground_truths),
            **STaDSMetrics.integrity_metrics(predictions, ground_truths)
        }
    
    @staticmethod
    def lao_per_feature_drop(
        baseline_score: float,
        ablated_scores: Sequence[float],
    ) -> List[float]:
        """
        Leave-Any-Out (LAO) attribution scores.

        Given:
            - baseline_score: e.g. penalised_accuracy with all features present
            - ablated_scores: list/array of the same metric when each feature
            is ablated (one feature per entry, aligned with feature order)

        Returns
        -------
        drops : list[float]
            baseline_score - ablated_score for each feature.

        Interpretation:
            Larger positive values => model performance relies more strongly
            on that feature.
        """
        return [baseline_score - s for s in ablated_scores]

    @staticmethod
    def self_attribution_recall(
        self_ranking: Sequence[Any],
        gt_feature_set: Sequence[Any],
        top_k: int | None = None,
    ) -> float:
        """
        Percentage of ground-truth important features covered by the model’s
        self-reported important features.
        """
        gt_set = set(gt_feature_set)
        if not gt_set:
            return 1.0

        if top_k is None:
            top_k = len(gt_set)
        top_k = max(0, min(len(self_ranking), top_k))

        predicted_set = set(self_ranking[:top_k])
        hits = len(predicted_set & gt_set)
        return hits / len(gt_set)

    @staticmethod
    def _rank_from_scores(
        scores: Mapping[Any, float],
        higher_is_better: bool = True,
    ) -> Dict[Any, float]:
        """
        Convert a dict {item -> score} into {item -> rank}.
        """
        if not scores:
            return {}

        items = list(scores.items())
        items.sort(key=lambda x: x[1], reverse=higher_is_better)

        ranks: Dict[Any, float] = {}
        i = 0
        n = len(items)
        while i < n:
            j = i + 1
            while j < n and items[j][1] == items[i][1]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0  # 1-based average rank
            for k in range(i, j):
                ranks[items[k][0]] = avg_rank
            i = j
        return ranks

    @staticmethod
    def self_faith(
        self_ranking: Sequence[Any],
        lao_scores: Mapping[Any, float],
        col_names_dir: Dict[Any, Any],
    ) -> float:
        """
        Self-Faith: agreement between self-claimed attribution ranking and
        behaviour-based LAO ranking.

        - self_ranking: feature IDs ordered from most to least important.
        - lao_scores: {feature -> LAO drop}, higher = stronger reliance.
        """
        feat_set = list(col_names_dir.values())
        if not feat_set:
            return float("nan")

        # self ranks
        feature_to_idx = {v: k for k, v in col_names_dir.items()}
        r_self = [feature_to_idx[feat] for feat in self_ranking]
        # LAO ranks: higher drop = more important
        lao_rank = STaDSMetrics._rank_from_scores(lao_scores, higher_is_better=True)
        r_lao = list(lao_rank.keys())
        rho, p = spearmanr(r_self, r_lao)
        return rho, p

    @staticmethod
    def lao_magnitude(
        lao_scores: Mapping[Any, float],
        method: str = "std",
    ) -> float:
        """
        LAO Magnitude: dispersion of behavioural reliance across features.

        Given a dict {feature -> LAO_drop}, we compute a scalar describing how
        concentrated or uniform the reliance is.

        method:
        - "std": standard deviation of normalised scores.
        """
        if not lao_scores:
            return float("nan")

        vals = [max(0.0, float(v)) for v in lao_scores.values()]
        total = sum(vals)
        if total <= 0:
            return float("nan")
        p = [v / total for v in vals]
        mean = sum(p) / len(p)
        return math.sqrt(sum((x - mean) ** 2 for x in p) / len(p))
       