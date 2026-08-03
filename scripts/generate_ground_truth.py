import logging

from santa_pola_rag.evaluation.ground_truth import (
    generate_ground_truth,
    save_ground_truth,
)

OUTPUT_PATH = "eval/ground_truth.json"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    items = generate_ground_truth(n_samples=30)
    save_ground_truth(items, OUTPUT_PATH)
    print(f"Saved {len(items)} ground truth questions to {OUTPUT_PATH}")
