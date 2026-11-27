"""
Evaluates student academic performance and ML model metrics from mathverse.db.

Run from backend/:
    python tools/evaluate_metrics.py
"""
import math
import sqlite3
from pathlib import Path

# BKT baseline parameters from services/mastery.py
P_SLIP = 0.10
P_GUESS = 0.25

DB_PATH = Path(__file__).resolve().parent.parent / "mathverse.db"


def evaluate():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    attempts = cur.execute(
        "SELECT * FROM attempts WHERE answered_at IS NOT NULL"
    ).fetchall()
    total_answered = len(attempts)

    if total_answered == 0:
        print("No answered attempts found in database.")
        return

    print("=" * 60)
    print(" MATHVERSE AI — ACADEMIC & ML EVALUATION METRICS REPORT")
    print("=" * 60)

    # 1. Student Academic Accuracy
    correct_count = sum(1 for a in attempts if a["is_correct"])
    overall_acc = (correct_count / total_answered) * 100
    avg_response_s = (
        sum(a["response_time_ms"] or 0 for a in attempts) / (total_answered * 1000)
    )

    print("\n[1] STUDENT ACADEMIC METRICS")
    print(f"  • Total Answered Questions: {total_answered}")
    print(f"  • Total Correct Answers:   {correct_count}")
    print(f"  • Overall Student Accuracy: {overall_acc:.2f}%")
    print(f"  • Avg Response Time:        {avg_response_s:.2f} seconds")

    # Topic breakdown
    print("\n  Topic Breakdown:")
    topics = sorted(set(a["topic"] for a in attempts))
    for t in topics:
        t_attempts = [a for a in attempts if a["topic"] == t]
        t_correct = sum(1 for a in t_attempts if a["is_correct"])
        t_acc = (t_correct / len(t_attempts)) * 100
        print(f"    - {t:20s}: {t_acc:5.1f}% accuracy ({t_correct}/{len(t_attempts)})")

    # 2. Bayesian Knowledge Tracing (BKT) Model Predictive Performance
    # BKT predicts P(correct) = p * (1 - slip) + (1 - p) * guess
    bkt_eval = [
        a for a in attempts if a["mastery_before"] is not None and a["is_correct"] is not None
    ]

    print("\n[2] BKT MASTERY MODEL EVALUATION (Next-Item Prediction)")
    if bkt_eval:
        tp = fp = tn = fn = 0
        brier_sum = 0.0

        for a in bkt_eval:
            p = a["mastery_before"]
            y = 1 if a["is_correct"] else 0
            p_correct = p * (1 - P_SLIP) + (1 - p) * P_GUESS
            pred_y = 1 if p_correct >= 0.5 else 0

            brier_sum += (p_correct - y) ** 2
            if pred_y == 1 and y == 1:
                tp += 1
            elif pred_y == 1 and y == 0:
                fp += 1
            elif pred_y == 0 and y == 0:
                tn += 1
            else:
                fn += 1

        n_eval = len(bkt_eval)
        bkt_acc = ((tp + tn) / n_eval) * 100
        rmse = math.sqrt(brier_sum / n_eval)
        precision = (tp / (tp + fp) * 100) if (tp + fp) else 0.0
        recall = (tp / (tp + fn) * 100) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        print(f"  • Evaluated Next-Item Transitions: {n_eval}")
        print(f"  • BKT Prediction Accuracy:        {bkt_acc:.2f}% (Threshold = 0.50)")
        print(f"  • Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  • Precision:                      {precision:.2f}%")
        print(f"  • Recall:                         {recall:.2f}%")
        print(f"  • F1-Score:                       {f1:.2f}%")
        print(f"  • Confusion Matrix:")
        print(f"      [ True Positive (TP): {tp:3d} | False Positive (FP): {fp:3d} ]")
        print(f"      [ False Negative(FN): {fn:3d} | True Negative  (TN): {tn:3d} ]")
    else:
        print("  No attempts with mastery_before data available.")

    # 3. Attention Tracker Metrics
    measured_att = [a for a in attempts if a["on_screen_pct"] is not None]
    print("\n[3] ATTENTION ESTIMATE METRICS (MediaPipe In-Browser)")
    print(f"  • Questions With Camera Active:    {len(measured_att)} / {total_answered} ({100*len(measured_att)/total_answered:.1f}%)")
    print(f"  • Questions With Camera Off (NULL): {total_answered - len(measured_att)}")

    if measured_att:
        avg_face = sum(a["face_present_pct"] or 0 for a in measured_att) / len(measured_att)
        avg_screen = sum(a["on_screen_pct"] or 0 for a in measured_att) / len(measured_att)
        blinks = [a["blink_rate_per_min"] for a in measured_att if a["blink_rate_per_min"] is not None]
        avg_blinks = sum(blinks) / len(blinks) if blinks else 0.0

        print(f"  • Avg Face Presence:               {avg_face:.1f}%")
        print(f"  • Avg On-Screen Time:              {avg_screen:.1f}%")
        print(f"  • Avg Blink Rate:                  {avg_blinks:.1f} blinks/min (measured across {len(blinks)} attempts)")

        # Label counts
        labels = {}
        for a in measured_att:
            lbl = a["attention_label"] or "Unknown"
            labels[lbl] = labels.get(lbl, 0) + 1
        print("  • Attention State Distribution:")
        for lbl, cnt in sorted(labels.items()):
            print(f"      - {lbl:15s}: {cnt:3d} ({100*cnt/len(measured_att):.1f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    evaluate()
