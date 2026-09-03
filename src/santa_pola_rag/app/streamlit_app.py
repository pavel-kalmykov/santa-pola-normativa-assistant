import json
import logging
import time
import uuid
from datetime import UTC, datetime

import streamlit as st
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessagesTypeAdapter

from santa_pola_rag.app.i18n import AVAILABLE_LANGUAGES, strings_for
from santa_pola_rag.app.suggestions import suggested_questions
from santa_pola_rag.config import settings
from santa_pola_rag.language import detect_language
from santa_pola_rag.observability import chat_store
from santa_pola_rag.observability.feedback import (
    ensure_index as ensure_feedback_index,
)
from santa_pola_rag.observability.feedback import record_feedback
from santa_pola_rag.observability.query_log import (
    ensure_index as ensure_query_log_index,
)
from santa_pola_rag.observability.query_log import queries_today, record_query
from santa_pola_rag.observability.tracing import setup_tracing
from santa_pola_rag.rag.agent import generate_title, stream_ask
from santa_pola_rag.rag.citations import render_footnotes
from santa_pola_rag.search.hybrid import RetrievalUnavailableError

# Unlike the CLI entry points (ingestion/pipeline.py, indexing/build_index.py),
# nothing configured the root logger's level here, so every logger.info() call
# in the whole app, including query_log.py's own success-path logging, was
# silently dropped: Python's root logger defaults to WARNING.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared between the live-generation path and the display_messages replay
# loop, so a failed turn renders identically (same message, same icon)
# whether it just happened or is being redrawn after a later st.rerun().
_ERROR_ICONS = {
    "retrieval_unavailable": ":material/cloud_off:",
    "llm_unavailable": ":material/block:",
    "other": ":material/error:",
}

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
    page_icon=":material/account_balance:",
)

setup_tracing()
ensure_feedback_index()
ensure_query_log_index()
chat_store.ensure_table()

# A cookie survives closing the tab/browser, which st.session_state does
# not; st.query_params is what actually makes that value visible to THIS
# script run without waiting for a reconnect, since Streamlit has no
# server-side "set a cookie" API of its own (only the read-only
# st.context.cookies). First-ever visit: generate a browser id, put it in
# both places. Returning visit with the cookie but no query param (a fresh
# tab): copy the cookie into query_params so the rest of this run can rely
# on one source of truth.
if "bid" in st.query_params:
    browser_id = st.query_params["bid"]
elif "sp_bid" in st.context.cookies:
    browser_id = st.context.cookies["sp_bid"]
    st.query_params["bid"] = browser_id
else:
    browser_id = str(uuid.uuid4())
    st.query_params["bid"] = browser_id
    st.html(
        f'<script>document.cookie = "sp_bid={browser_id}; max-age=31536000; '
        'path=/; SameSite=Lax";</script>',
        unsafe_allow_javascript=True,
    )

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "pydantic_history" not in st.session_state:
    st.session_state.pydantic_history = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "chat_title" not in st.session_state:
    st.session_state.chat_title = None
if "chat_created_at" not in st.session_state:
    st.session_state.chat_created_at = None

strings = strings_for(_current_ui_language())


def _start_new_chat() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.pydantic_history = []
    st.session_state.display_messages = []
    st.session_state.chat_title = None
    st.session_state.chat_created_at = None


with st.sidebar:
    # Sidebar, not the header row next to the title: every chat product
    # (ChatGPT, Claude, Open WebUI) puts conversation-management controls
    # there, not floating in the main content area, and it's where the list
    # of past chats for this browser lives too.
    past_chats = chat_store.list_chats(browser_id)
    if not past_chats:
        st.caption(strings["no_chats_yet"])
    else:
        for chat in past_chats:
            sid = chat["session_id"]
            is_active = sid == st.session_state.session_id
            renaming_key = f"renaming_{sid}"
            deleting_key = f"deleting_{sid}"

            if st.session_state.get(renaming_key):
                new_title = st.text_input(
                    strings["rename_label"],
                    value=chat["title"],
                    key=f"rename_input_{sid}",
                    label_visibility="collapsed",
                )
                with st.container(horizontal=True, gap="small"):
                    if st.button(
                        strings["rename_confirm"],
                        key=f"rename_save_{sid}",
                        type="primary",
                        width="stretch",
                    ):
                        chat_store.rename_chat(sid, new_title)
                        if is_active:
                            st.session_state.chat_title = new_title
                        del st.session_state[renaming_key]
                        st.rerun()
                    if st.button(
                        strings["rename_cancel"], key=f"rename_cancel_{sid}", width="stretch"
                    ):
                        del st.session_state[renaming_key]
                        st.rerun()
            elif st.session_state.get(deleting_key):
                st.caption(strings["confirm_delete"])
                with st.container(horizontal=True, gap="small"):
                    if st.button(
                        strings["delete_confirm_yes"],
                        key=f"delete_yes_{sid}",
                        type="primary",
                        width="stretch",
                    ):
                        chat_store.delete_chat(sid)
                        if is_active:
                            _start_new_chat()
                        del st.session_state[deleting_key]
                        st.rerun()
                    if st.button(
                        strings["delete_confirm_cancel"],
                        key=f"delete_cancel_{sid}",
                        width="stretch",
                    ):
                        del st.session_state[deleting_key]
                        st.rerun()
            else:
                # st.columns, not st.container(horizontal=True): a flex
                # container lets a wide-enough title button's own intrinsic
                # content width push the sibling menu button onto a second
                # line entirely, since flex items don't shrink below their
                # content's natural width by default. Columns instead give
                # each side a hard, pre-allocated percentage of the row, so
                # a long title wraps within its own column instead of
                # squeezing its sibling out.
                title_col, menu_col = st.columns([0.82, 0.18])
                with title_col:
                    if st.button(
                        chat["title"],
                        key=f"chat_{sid}",
                        width="stretch",
                        type="secondary" if is_active else "tertiary",
                        disabled=is_active,
                    ):
                        loaded = chat_store.load_chat(sid)
                        if loaded is not None:
                            st.session_state.session_id = loaded["session_id"]
                            st.session_state.chat_title = loaded["title"]
                            st.session_state.chat_created_at = loaded["created_at"]
                            st.session_state.display_messages = json.loads(
                                loaded["display_messages_json"]
                            )
                            st.session_state.pydantic_history = (
                                ModelMessagesTypeAdapter.validate_json(
                                    loaded["pydantic_history_json"]
                                )
                            )
                            st.rerun()
                with menu_col:
                    # A single "more options" menu, not two always-visible
                    # icon buttons: two extra icons crammed into an already
                    # narrow sidebar row left less room for the title and
                    # read as cluttered (matches the pattern used by Claude
                    # Desktop's own chat list).
                    with st.popover(":material/more_vert:"):
                        if st.button(
                            strings["rename_help"],
                            icon=":material/edit:",
                            key=f"rename_btn_{sid}",
                            width="stretch",
                        ):
                            st.session_state[renaming_key] = True
                            st.rerun()
                        if st.button(
                            strings["delete_help"],
                            icon=":material/delete:",
                            key=f"delete_btn_{sid}",
                            width="stretch",
                        ):
                            st.session_state[deleting_key] = True
                            st.rerun()

    # A full-width button with a real label, not an icon-only tertiary one
    # (a bare icon was not intuitive enough to communicate "start a new
    # chat"), placed after the chat list so it always sits at the bottom of
    # the sidebar instead of floating above a list that grows underneath it.
    if st.button(
        strings["new_chat_help"],
        icon=":material/edit_square:",
        width="stretch",
        key="new_chat",
    ):
        _start_new_chat()
        st.rerun()

title_col, lang_col = st.columns([5, 2], vertical_alignment="center")
with lang_col:
    with st.popover(f":material/language: {st.session_state.ui_language_label}"):
        # A fixed, bilingual label instead of one translated into the
        # currently selected language: reusing this widget's own selection
        # to translate its own label is self-referential and, combined with
        # Streamlit keying widgets by label when no `key` is given, was
        # exactly what made picking a language take two clicks instead of
        # one (the label changed, so Streamlit treated it as a brand new
        # widget on the very next rerun and forgot the choice). `key=`
        # binds it to session_state directly instead, which is the
        # supported way to persist a widget's value across reruns. A radio
        # group, not a selectbox: the popover's own trigger already looks
        # like a dropdown (Streamlit adds its own chevron), so a selectbox
        # nested inside it read as a dropdown-to-open-a-dropdown.
        st.radio(
            "Language / Idioma",
            options=list(AVAILABLE_LANGUAGES.values()),
            key="ui_language_label",
        )

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
        st.caption(f":material/warning: {r['message']}")
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


def _error_message(strings: dict, error_type: str) -> str:
    return strings["error"] if error_type == "other" else strings[error_type]


def _chat_title(question: str) -> str:
    question = question.strip()
    return question if len(question) <= 60 else question[:57] + "..."


for index, message in enumerate(st.session_state.display_messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "blocks" in message:
            render_saved_blocks(
                message["blocks"], strings, message.get("question_language"), index
            )
            error_type = message.get("error_type")
            if error_type is not None:
                st.error(
                    _error_message(strings, error_type), icon=_ERROR_ICONS[error_type]
                )
            if message.get("text_search_degraded"):
                st.info(strings["search_degraded"], icon=":material/manage_search:")
            if message.get("vector_search_degraded"):
                st.info(strings["vector_search_degraded"], icon=":material/psychology:")
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant":
            is_last_message = index == len(st.session_state.display_messages) - 1
            feedback = message.get("feedback")
            # st.columns sizes by ratio, so even a "gap=0" pair of narrow
            # columns leaves each button centered inside its own wide
            # slot; a content-width horizontal container is what actually
            # packs the icons tight against each other.
            with st.container(horizontal=True, gap="small", width="content"):
                if st.button(
                    "",
                    icon=":material/content_copy:",
                    type="tertiary",
                    help=strings["copy_help"],
                    key=f"copy_{index}",
                ):
                    st.session_state.copy_pending = message["content"]
                    st.rerun()
                # Regenerating an earlier message would need to discard every
                # message that came after it too, so this is only offered on
                # the latest turn.
                if is_last_message and "history_before" in message:
                    if st.button(
                        "",
                        icon=":material/refresh:",
                        type="tertiary",
                        help=strings["regenerate_help"],
                        key=f"regen_{index}",
                    ):
                        st.session_state.pydantic_history = message["history_before"]
                        st.session_state.display_messages.pop()
                        st.session_state.regenerate_pending = message["question"]
                        st.rerun()
                if feedback is None:
                    if st.button(
                        "",
                        icon=":material/thumb_up:",
                        type="tertiary",
                        help=strings["feedback_up_help"],
                        key=f"up_{index}",
                    ):
                        record_feedback(
                            st.session_state.session_id,
                            message["question"],
                            message["content"],
                            1,
                            message.get("question_language"),
                        )
                        message["feedback"] = 1
                        st.rerun()
                    if st.button(
                        "",
                        icon=":material/thumb_down:",
                        type="tertiary",
                        help=strings["feedback_down_help"],
                        key=f"down_{index}",
                    ):
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
                if "created_at" in message:
                    # Static markup, no script involved, so it always
                    # reflects the current message on every render; the
                    # actual relative-time text is filled in by the shared
                    # formatter script below. The sibling buttons are 40px
                    # tall flex boxes centering their own icon; a bare
                    # <time> is just inline text with no box of its own, so
                    # without a same-height wrapper its text sits a few
                    # pixels above the buttons' vertical center instead of
                    # lining up with them.
                    st.html(
                        '<span style="display:inline-flex; align-items:center; '
                        'height:40px;">'
                        f'<time class="sp-timestamp" '
                        f'datetime="{message["created_at"]}" '
                        f'data-locale="{_current_ui_language()}" '
                        'style="opacity:0.6; font-size:0.8rem;"></time>'
                        "</span>"
                    )

# Intl.RelativeTimeFormat only formats a (value, unit) pair once, on demand;
# it doesn't track the original timestamp or re-format itself as time
# passes, so this recomputes every sp-timestamp element's text against the
# current time on every script rerun (the nonce keeps the injected <script>
# body unique so it actually re-executes every time, not just once: see
# copy_pending's own comment above for the same st.html memoization gotcha).
st.html(
    f"""<script>/*{uuid.uuid4()}*/
(function() {{
    var units = [["year", 31536000], ["month", 2592000], ["day", 86400],
                 ["hour", 3600], ["minute", 60], ["second", 1]];
    document.querySelectorAll("time.sp-timestamp").forEach(function(el) {{
        var date = new Date(el.getAttribute("datetime"));
        var locale = el.getAttribute("data-locale");
        var diffSeconds = (date.getTime() - Date.now()) / 1000;
        var value = Math.round(diffSeconds);
        var unit = "second";
        for (var i = 0; i < units.length; i++) {{
            // Rounds before comparing against the threshold, not after:
            // otherwise a diff of e.g. 59.8s stays in the "second" bucket
            // (59.8 < 60) but then rounds up to display as "60 seconds
            // ago" instead of crossing over into "1 minute ago".
            var rounded = Math.round(diffSeconds / units[i][1]);
            if (Math.abs(rounded) >= 1) {{
                value = rounded;
                unit = units[i][0];
                break;
            }}
        }}
        var rtf = new Intl.RelativeTimeFormat(locale, {{numeric: "auto"}});
        el.textContent = rtf.format(value, unit);
        el.title = new Intl.DateTimeFormat(
            locale, {{dateStyle: "medium", timeStyle: "short"}}
        ).format(date);
    }});
}})();
</script>""",
    unsafe_allow_javascript=True,
)

copy_text = st.session_state.pop("copy_pending", None)
if copy_text is not None:
    # navigator.clipboard.writeText must run in the browser; the server and
    # the browser clipboard are different machines in a real deployment, so
    # this can't be done from Python alone (e.g. pyperclip). The nonce
    # comment keeps the injected <script> body unique on every click: with
    # byte-identical body text (copying the same answer twice in a row),
    # Streamlit's st.html memoizes on that string and skips re-running the
    # script the second time, so nothing gets copied.
    st.html(
        f"<script>/*{uuid.uuid4()}*/navigator.clipboard.writeText"
        f"({json.dumps(copy_text)});</script>",
        unsafe_allow_javascript=True,
    )
    st.toast(strings["copied_toast"], icon=":material/content_copy:")

clicked_suggestion = None
if not st.session_state.display_messages:
    suggestions = suggested_questions(_current_ui_language())
    if suggestions:
        st.caption(strings["suggestions_label"])
        for suggestion in suggestions:
            if st.button(
                suggestion, key=f"suggestion_{suggestion}", use_container_width=True
            ):
                clicked_suggestion = suggestion

# Consumed first, and independent of the chat_input/suggestion sources
# below, so the click handler above can trigger a regeneration without a
# duplicate user bubble (the original question is still in display_messages;
# only the stale assistant answer was popped).
regenerate_pending = st.session_state.pop("regenerate_pending", None)
is_regenerate = regenerate_pending is not None
question = regenerate_pending or st.chat_input(strings["chat_placeholder"]) or clicked_suggestion


@st.cache_data(ttl=60)
def _queries_today_cached() -> int | None:
    # No proxy-level rate limiting exists on Streamlit Community Cloud, so
    # the spend guard is these two app-side checks: a per-session cap from
    # session_state and this global daily ceiling from the query log. The
    # cache keeps the ceiling check from turning every question into an
    # extra OpenSearch round trip.
    return queries_today()


if question:
    asked = st.session_state.get("questions_asked", 0)
    if asked >= settings.max_questions_per_session:
        st.error(
            strings["rate_limit_session"].format(
                n=settings.max_questions_per_session
            ),
            icon=":material/block:",
        )
        st.stop()
    today_count = _queries_today_cached()
    if today_count is not None and today_count >= settings.daily_query_budget:
        st.error(strings["rate_limit_daily"], icon=":material/block:")
        st.stop()
    st.session_state.questions_asked = asked + 1

    if not is_regenerate:
        st.session_state.display_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

    detected_language = detect_language(question)
    question_language = detected_language[0] if detected_language else None
    message_id = len(st.session_state.display_messages)
    history_before = st.session_state.pydantic_history

    with st.chat_message("assistant"):
        start_time = time.monotonic()
        stream, holder = stream_ask(question, message_history=history_before)
        # The agent's own narration before each tool call ("I'll search
        # for...") is shown too, not hidden: it's rendered live as italic
        # text, one block per text run, with a persistent card for every
        # tool call (query + the actual results found). Only once the whole
        # run finishes do we know which text block was the final answer
        # (whichever one has no tool call after it) and restyle it plainly.
        blocks: list[dict] = []

        def render_narration(block: dict) -> None:
            block["placeholder"].markdown(f"*{block['text']}*")

        error_type = None
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
        except RetrievalUnavailableError:
            logger.exception("Retrieval backend unavailable for question: %s", question)
            error_type = "retrieval_unavailable"
        except ModelAPIError:
            logger.exception("LLM provider rejected the request: %s", question)
            error_type = "llm_unavailable"
        except Exception:
            logger.exception("Agent call failed for question: %s", question)
            error_type = "other"

        # Whatever narration/tool calls happened before a mid-stream failure
        # stay visible (real diagnostic signal: "it tried X, then broke"),
        # with the error rendered right after them, not in place of them.
        # The degradation notice itself is rendered from display_messages on
        # the next script run (see the replay loop above), not here:
        # st.rerun() below re-executes the whole script immediately, and
        # anything drawn only in this live block, not saved into the message
        # dict, would otherwise vanish the instant the response finishes
        # (same reasoning as render_saved_blocks's own docstring).
        text_blocks = [b for b in blocks if b["type"] == "text"]
        final_block = text_blocks[-1] if text_blocks else None
        if error_type is None and final_block is not None:
            final_block["placeholder"].markdown(
                render_footnotes(final_block["text"], message_id, question_language),
                unsafe_allow_html=True,
            )
        if error_type is not None:
            st.error(_error_message(strings, error_type), icon=_ERROR_ICONS[error_type])
            answer = _error_message(strings, error_type)
        else:
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
        # holder.messages only gets populated on a successful run (see
        # agent.py's worker()); overwriting the conversation's memory with
        # None on a failed turn would lose it for the next question.
        if holder.messages is not None:
            st.session_state.pydantic_history = holder.messages

    record_query(
        st.session_state.session_id,
        message_id,
        question,
        answer,
        question_language,
        latency_ms,
        search_time_ms=int(holder.search_time_ms),
        error_type=error_type,
    )
    st.session_state.display_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "blocks": blocks_data,
            "question": question,
            "question_language": question_language,
            "feedback": None,
            "text_search_degraded": holder.text_search_degraded,
            "vector_search_degraded": holder.vector_search_degraded,
            "error_type": error_type,
            "history_before": history_before,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )

    if st.session_state.chat_title is None:
        title = None
        if error_type is None:
            try:
                title = generate_title(question, answer)
            except Exception:
                logger.exception("Title generation failed for question: %s", question)
        st.session_state.chat_title = title or _chat_title(question)
        st.session_state.chat_created_at = datetime.now(UTC).isoformat()
    # history_before holds raw pydantic-ai message objects (needed live for
    # the regenerate button), not JSON-serializable on their own; dropped
    # here since pydantic_history below already carries the full history
    # for resuming a saved chat.
    serializable_messages = [
        {k: v for k, v in m.items() if k != "history_before"}
        for m in st.session_state.display_messages
    ]
    chat_store.save_chat(
        browser_id,
        st.session_state.session_id,
        st.session_state.chat_title,
        st.session_state.chat_created_at,
        json.dumps(serializable_messages),
        ModelMessagesTypeAdapter.dump_json(st.session_state.pydantic_history).decode(),
    )
    st.rerun()
