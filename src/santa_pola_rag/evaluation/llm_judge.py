import logging

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from santa_pola_rag.config import settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_HTTP_CLIENT = httpx.AsyncClient(timeout=60.0)
# Judging with a different model/provider than the one that generated the
# answer (DeepSeek) avoids the model favoring its own phrasing/reasoning.
JUDGE_MODEL = "google/gemini-2.5-flash"


class JudgeVerdict(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning before the verdict")
    relevance: int = Field(
        ge=1, le=5, description="Does the answer address the question?"
    )
    faithfulness: int = Field(
        ge=1, le=5, description="Is the answer grounded in the retrieved context?"
    )
    cites_source: bool = Field(description="Does the answer cite a source document?")
    passed: bool = Field(description="Overall pass/fail verdict")


JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator of a RAG assistant for Santa Pola's municipal \
ordinances. Given a user question, the retrieved context excerpts and the \
assistant's answer, reason step by step about:
1. Relevance: does the answer actually address the question? (1-5)
2. Faithfulness: is every claim in the answer supported by the retrieved \
   context, with nothing invented? (1-5)
3. Citation: does the answer cite a source document (title/page/URL)?

Only pass an answer (passed=true) if relevance >= 4, faithfulness >= 4 and \
cites_source is true.
"""

_model = OpenAIChatModel(
    JUDGE_MODEL,
    provider=OpenAIProvider(
        base_url=OPENROUTER_BASE_URL,
        api_key=settings.openrouter_api_key,
        http_client=_HTTP_CLIENT,
    ),
)
_judge_agent = Agent(
    model=_model, output_type=JudgeVerdict, system_prompt=JUDGE_SYSTEM_PROMPT
)


def judge_answer(question: str, answer: str, retrieved_context: str) -> JudgeVerdict:
    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{retrieved_context}\n\n"
        f"Assistant's answer:\n{answer}"
    )
    result = _judge_agent.run_sync(prompt)
    return result.output
