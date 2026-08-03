import json
import logging

from santa_pola_rag.evaluation.ground_truth import load_ground_truth
from santa_pola_rag.evaluation.retrieval_eval import evaluate_all

GROUND_TRUTH_PATH = "eval/ground_truth.json"
OUTPUT_PATH = "eval/retrieval_results.json"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    results = evaluate_all(ground_truth, top_k=5)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for name, metrics in results.items():
        print(f"{name:12s} hit_rate={metrics['hit_rate']:.3f} mrr={metrics['mrr']:.3f}")
