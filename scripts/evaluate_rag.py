import json
import logging

from santa_pola_rag.evaluation.ground_truth import load_ground_truth
from santa_pola_rag.evaluation.llm_judge import judge_answer
from santa_pola_rag.rag.agent import ask
from santa_pola_rag.search.hybrid import hybrid_search

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = "eval/ground_truth.json"
OUTPUT_PATH = "eval/rag_judge_results.json"
N_SAMPLES = 10


def run() -> list[dict]:
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)[:N_SAMPLES]
    results = []

    for item in ground_truth:
        answer = ask(item.question).output
        context_chunks = hybrid_search(item.question, top_k=5)
        context = "\n---\n".join(c.text for c in context_chunks)

        verdict = judge_answer(item.question, answer, context)
        results.append(
            {
                "question": item.question,
                "answer": answer,
                "verdict": verdict.model_dump(),
            }
        )
        logger.info(
            "passed=%s relevance=%d faithfulness=%d cites_source=%s: %s",
            verdict.passed,
            verdict.relevance,
            verdict.faithfulness,
            verdict.cites_source,
            item.question[:60],
        )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n_passed = sum(1 for r in results if r["verdict"]["passed"])
    print(f"\n{n_passed}/{len(results)} answers passed LLM-as-judge evaluation")
