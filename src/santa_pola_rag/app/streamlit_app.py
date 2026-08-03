import logging
import time
import uuid

import streamlit as st

from santa_pola_rag.app.i18n import AVAILABLE_LANGUAGES, strings_for
from santa_pola_rag.language import detect_language
from santa_pola_rag.observability.feedback import (
    ensure_index as ensure_feedback_index,
)
from santa_pola_rag.observability.feedback import record_feedback
from santa_pola_rag.observability.query_log import (
    ensure_index as ensure_query_log_index,
)
from santa_pola_rag.observability.query_log import record_query
from santa_pola_rag.observability.tracing import setup_tracing
from santa_pola_rag.rag.agent import stream_ask
from santa_pola_rag.rag.citations import render_footnotes

logger = logging.getLogger(__name__)

if "ui_language_label" not in st.session_state:
    st.session_state.ui_language_label = AVAILABLE_LANGUAGES["es"]


def _current_ui_language() -> str:
    return next(
        code
        for code, label in AVAILABLE_LANGUAGES.items()
        if label == st.session_state.ui_language_label
    )


st.set_page_config(
    page_title=strings_for(_current_ui_language())["page_title"],
    page_icon="🏛️",
)

setup_tracing()
ensure_feedback_index()
ensure_query_log_index()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "pydantic_history" not in st.session_state:
    st.session_state.pydantic_history = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

title_col, lang_col = st.columns([6, 1])
with lang_col:
    with st.popover("🌐"):
        # A fixed, bilingual label instead of one translated into the
        # currently selected language: reusing this widget's own selection
        # to translate its own label is self-referential and, combined with
        # Streamlit keying widgets by label when no `key` is given, was
        # exactly what made picking a language take two clicks instead of
        # one (the label changed, so Streamlit treated it as a brand new
        # widget on the very next rerun and forgot the choice). `key=`
        # binds it to session_state directly instead, which is the
        # supported way to persist a widget's value across reruns.
        st.selectbox(
            "Language / Idioma",
            options=list(AVAILABLE_LANGUAGES.values()),
            key="ui_language_label",
        )

strings = strings_for(_current_ui_language())

with title_col:
    st.title(strings["heading"], anchor=False)
st.caption(strings["caption"])
with st.expander(strings["about_title"]):
    st.markdown(strings["about_body"])


def _render_result_line(r: dict) -> None:
    # search_ordinances returns a single {"error": ...} dict instead of the
    # usual title/page_number/url shape once the per-question search budget
    # (see agent.py's MAX_SEARCHES_PER_TURN) runs out, so this can't assume
    # every entry looks like a real search hit.
    if "error" in r:
        st.caption(f"⚠️ {r['message']}")
        return
    st.markdown(
        f"- **{r['title']}** (p. {r['page_number']}) - [{r['url']}]({r['url']})"
    )


def render_saved_blocks(
    blocks_data: list[dict], strings: dict, answer_language: str | None, message_id: int
) -> None:
    """Replay a persisted trace (narration + tool-call cards + final answer)
    exactly as it looked live. st.rerun() re-executes the whole script after
    every response, and st.chat_input() only returns the new question once;
    on that rerun the live-generation code below never runs again, so
    anything not saved here (the trace used to only save the final answer
    text) visibly vanished the instant the response finished."""
    for block in blocks_data:
        if block["type"] == "text":
            if block.get("final"):
                st.markdown(
                    render_footnotes(block["text"], message_id, answer_language),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"*{block['text']}*")
        elif block["type"] == "tool":
            st.caption(strings["searching"].format(query=block["query"]))
            results = block.get("results")
            if results:
                with st.expander(strings["results_found"].format(n=len(results))):
                    for r in results:
                        _render_result_line(r)
            elif results == []:
                st.caption(strings["no_results"])


for index, message in enumerate(st.session_state.display_messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "blocks" in message:
            render_saved_blocks(
                message["blocks"], strings, message.get("question_language"), index
            )
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant":
            col_up, col_down, _ = st.columns([1, 1, 8])
            feedback = message.get("feedback")
            if feedback is None:
                if col_up.button("👍", key=f"up_{index}"):
                    record_feedback(
                        st.session_state.session_id,
                        message["question"],
                        message["content"],
                        1,
                        message.get("question_language"),
                    )
                    message["feedback"] = 1
                    st.rerun()
                if col_down.button("👎", key=f"down_{index}"):
                    record_feedback(
                        st.session_state.session_id,
                        message["question"],
                        message["content"],
                        -1,
                        message.get("question_language"),
                    )
                    message["feedback"] = -1
                    st.rerun()
            else:
                st.caption(
                    strings["feedback_up"]
                    if feedback == 1
                    else strings["feedback_down"]
                )

question = st.chat_input(strings["chat_placeholder"])

if question:
    st.session_state.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    detected_language = detect_language(question)
    question_language = detected_language[0] if detected_language else None
    message_id = len(st.session_state.display_messages)

    with st.chat_message("assistant"):
        start_time = time.monotonic()
        stream, holder = stream_ask(
            question, message_history=st.session_state.pydantic_history
        )
        # The agent's own narration before each tool call ("I'll search
        # for...") is shown too, not hidden: it's rendered live as italic
        # text, one block per text run, with a persistent card for every
        # tool call (query + the actual results found). Only once the whole
        # run finishes do we know which text block was the final answer
        # (whichever one has no tool call after it) and restyle it plainly.
        blocks: list[dict] = []

        def render_narration(block: dict) -> None:
            block["placeholder"].markdown(f"*{block['text']}*")

        try:
            for event in stream:
                if event.kind == "text":
                    if blocks and blocks[-1]["type"] == "text":
                        blocks[-1]["text"] += event.value
                    else:
                        blocks.append(
                            {
                                "type": "text",
                                "placeholder": st.empty(),
                                "text": event.value,
                            }
                        )
                    render_narration(blocks[-1])
                elif event.kind == "tool_call":
                    placeholder = st.empty()
                    with placeholder.container():
                        st.caption(strings["searching"].format(query=event.value))
                    blocks.append(
                        {
                            "type": "tool",
                            "placeholder": placeholder,
                            "query": event.value,
                            "results": None,
                        }
                    )
                elif event.kind == "tool_result":
                    for block in reversed(blocks):
                        if block["type"] == "tool" and block["results"] is None:
                            block["results"] = event.value
                            with block["placeholder"].container():
                                st.caption(
                                    strings["searching"].format(query=block["query"])
                                )
                                if event.value:
                                    with st.expander(
                                        strings["results_found"].format(
                                            n=len(event.value)
                                        )
                                    ):
                                        for r in event.value:
                                            _render_result_line(r)
                                else:
                                    st.caption(strings["no_results"])
                            break
        except Exception:
            logger.exception("Agent call failed for question: %s", question)
            st.error(strings["error"])
            st.stop()

        text_blocks = [b for b in blocks if b["type"] == "text"]
        final_block = text_blocks[-1] if text_blocks else None
        if final_block is not None:
            final_block["placeholder"].markdown(
                render_footnotes(final_block["text"], message_id, question_language),
                unsafe_allow_html=True,
            )
        answer = final_block["text"] if final_block is not None else ""

        # Strip the (unpicklable, live-only) placeholders and record which
        # block was final, so the trace survives the st.rerun() below.
        blocks_data = [
            {"type": "text", "text": b["text"], "final": b is final_block}
            if b["type"] == "text"
            else {"type": "tool", "query": b["query"], "results": b["results"]}
            for b in blocks
        ]

        latency_ms = int((time.monotonic() - start_time) * 1000)
        st.session_state.pydantic_history = holder.messages

    record_query(
        st.session_state.session_id,
        question,
        answer,
        question_language,
        latency_ms,
        search_time_ms=int(holder.search_time_ms),
    )
    st.session_state.display_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "blocks": blocks_data,
            "question": question,
            "question_language": question_language,
            "feedback": None,
        }
    )
    st.rerun()
