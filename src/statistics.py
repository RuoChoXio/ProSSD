import json
import csv
import pathlib
import datetime
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, precision_score,
    recall_score, f1_score, accuracy_score, roc_auc_score
)

def get_optimal_threshold(human_scores, ai_scores):
    y_true = [0] * len(human_scores) + [1] * len(ai_scores)
    y_scores = list(human_scores) + list(ai_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = float(thresholds[optimal_idx])

    return optimal_threshold

def calculate_metrics(human_scores, ai_scores, threshold=None):
    y_true = [0] * len(human_scores) + [1] * len(ai_scores)
    y_scores = list(human_scores) + list(ai_scores)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    if threshold is None:
        optimal_idx = np.argmax(tpr - fpr)
        used_threshold = float(thresholds[optimal_idx])
        threshold_type = "optimal_youden (test_set_leakage)"
    else:
        used_threshold = float(threshold)
        threshold_type = "fixed (from_training)"

    y_pred = (y_scores >= used_threshold).astype(int)


    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tpr_at_1_fpr = np.interp(0.01, fpr, tpr)
    tpr_at_001_fpr = np.interp(0.0001, fpr, tpr)

    metrics = {
        "roc_auc": round(roc_auc * 100, 4),
        "accuracy": round(acc * 100, 4),
        "f1_score": round(f1 * 100, 4),
        "precision": round(precision * 100, 4),
        "recall": round(recall * 100, 4),
        "tpr_at_fpr_1_percent": round(tpr_at_1_fpr * 100, 4),
        "tpr_at_fpr_0_01_percent": round(tpr_at_001_fpr * 100, 4),
        "threshold": round(used_threshold, 6),
        "threshold_type": threshold_type,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        },
        "sample_counts": {
            "human": len(human_scores),
            "ai": len(ai_scores)
        }
    }

    return metrics

def save_results(metrics, config, output_dir):
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_name = pathlib.Path(config['test_data']).stem
    model_name = pathlib.Path(config['model_path']).name
    k = config['k']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    filename_base = f"stats_{model_name}_k{k}_{timestamp}"

    json_path = output_path / f"{filename_base}.json"
    full_record = {
        "config": config,
        "metrics": metrics,
        "timestamp": timestamp
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_record, f, indent=4)

    csv_path = output_path / "summary_results.csv"
    file_exists = csv_path.exists()

    csv_row = {
        "timestamp": timestamp,
        "model_name": model_name,
        "k": k,
        "dataset": dataset_name,
        "auc": metrics['roc_auc'],
        "acc": metrics['accuracy'],
        "f1": metrics['f1_score'],
        "tpr@0.01%fpr": metrics['tpr_at_fpr_0_01_percent'],
        "threshold": metrics['threshold'],
        "human_samples": metrics['sample_counts']['human'],
        "ai_samples": metrics['sample_counts']['ai']
    }

    fieldnames = list(csv_row.keys())

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_row)

    print(f"[Statistics] Detailed results saved to: {json_path}")
    print(f"[Statistics] Summary appended to: {csv_path}")
