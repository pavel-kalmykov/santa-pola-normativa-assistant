import json
import logging

from pydantic_ai.messages import ToolReturnPart

from santa_pola_rag.evaluation.ground_truth import load_ground_truth
from santa_pola_rag.evaluation.llm_judge import judge_answer
from santa_pola_rag.rag.agent import ask

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = "eval/ground_truth.json"
OUTPUT_PATH = "eval/rag_judge_results.json"
N_SAMPLES = 30


def _used_context(result) -> str:
    """The judge checks faithfulness against what the agent actually
    retrieved and grounded its answer in. A fresh hybrid_search(item.question)
    call is the wrong source for that: the agent always searches in Spanish
    regardless of the question's language (see agent.py's system prompt),
    so re-querying with the original, possibly non-Spanish question surfaces
    a different, often unrelated set of chunks and can fail a genuinely
    well-grounded answer for the wrong reason."""
    excerpts = []
    for message in result.all_messages():
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolReturnPart) and isinstance(part.content, list):
                for hit in part.content:
                    if isinstance(hit, dict) and "excerpt" in hit:
                        excerpts.append(hit["excerpt"])
    return "\n---\n".join(excerpts)


def run() -> list[dict]:
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)[:N_SAMPLES]
    results = []

    for item in ground_truth:
        result = ask(item.question)
        answer = result.output
        context = _used_context(result)

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
