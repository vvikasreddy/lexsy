import os
import re
import random
from collections import Counter
from typing import List, Tuple
import base64

import requests

import streamlit as st
from blank_filler import find_blanks_in_docx, replace_one_blank, DOCX_ENABLED as _DOCX_FILL_ENABLED
from answer_utils import extract_and_validate
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
                    pass

try:
    from PyPDF2 import PdfReader as _PdfReader
except Exception:
    _PdfReader = None

try:
    import docx as _docx
except Exception:
    _docx = None

st.set_page_config(page_title="Fill DOCX via Chat", layout="wide")

# Keep page scrollable even with many chat messages
try:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] .main { overflow-y: auto; }
        .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
except Exception:
    pass

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# State for DOCX filling
for k, v in {
    "docx_work_path": "",
    "fill_mode": False,
    "blanks": [],
    "pending_blank": None,
    "pending_confirmation": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

CHECKS_ENABLED = False


def _greeting() -> str:
    options = [
        "Hi! Let's fill your document together.",
        "Hello - ready to complete this form?",
        "Hey there! Shall we start filling the document?",
        "Welcome! Let's get your document filled in.",
        "Great to see you. Let's finish the form.",
    ]
    return random.choice(options)


def _is_yes(s: str) -> bool:
    try:
        t = (s or "").strip().lower()
    except Exception:
        return False
    return t in {
        "y", "yes", "yeah", "yep", "ok", "okay", "correct", "proceed",
        "go ahead", "looks good", "fine", "confirm", "sure",
    }


def _is_no(s: str) -> bool:
    try:
        t = (s or "").strip().lower()
    except Exception:
        return False
    return t in {"n", "no", "nope", "change", "edit", "retry", "different", "wrong", "incorrect"}


def _blank_like(val: str) -> bool:
    try:
        import re as _re
        return _re.fullmatch(r"\s*(\[\s*_{3,}\s*\]|_{3,})\s*", (val or "")) is not None
    except Exception:
        return False



def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith((".txt", ".md")):
        data = uploaded_file.read()
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode(errors="ignore")
    if name.endswith(".pdf"):
        if _PdfReader is None:
            return ""
        try:
            reader = _PdfReader(uploaded_file)
            parts = []
            for page in reader.pages:
                text = page.extract_text() or ""
                parts.append(text)
            return "\n".join(parts)
        except Exception:
            return ""
    if name.endswith(".docx"):
        if _docx is None:
            return ""
        try:
            doc = _docx.Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    return ""


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, 0)
    return [c for c in chunks if c.strip()]


def text_to_vector(text: str) -> Counter:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return Counter(tokens)


def cosine_sim(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    denom_a = sum(v * v for v in a.values()) ** 0.5
    denom_b = sum(v * v for v in b.values()) ** 0.5
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return num / (denom_a * denom_b)


def top_k_chunks(query: str, chunks: List[str], k: int = 3) -> List[Tuple[str, float]]:
    qv = text_to_vector(query)
    scored = [(c, cosine_sim(qv, text_to_vector(c))) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def generate_local_reply(prompt: str) -> str:
    if not st.session_state.doc_chunks:
        return (
            "I don't have a document loaded. You can upload one from the sidebar. "
            + "Meanwhile, here's your message back: "
            + prompt
        )
    top = top_k_chunks(prompt, st.session_state.doc_chunks, k=3)
    relevant = [c for c, s in top if s > 0]
    if not relevant:
        return (
            "I couldn't find relevant excerpts in the document. "
            "Try rephrasing or ask a different question."
        )
    joined = "\n\n".join(relevant)
    return "Here are relevant excerpts from your document:\n\n" + joined


def call_openai(messages: List[dict], context_chunks: List[str], api_key: str) -> str:
    ctx = "\n\n".join(context_chunks) if context_chunks else ""
    full_messages = []
    if ctx:
        full_messages.append(
            {
                "role": "system",
                "content": "Use the following document context when helpful:\n" + ctx,
            }
        )
    full_messages.extend(messages)
    try:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=full_messages,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            import openai

            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=full_messages,
                temperature=0.2,
            )
            return resp.choices[0].message["content"].strip()
    except Exception as e:
        return f"(OpenAI error) {e}"


def call_deepseek(messages: List[dict], context_chunks: List[str], api_key: str) -> str:
    ctx = "\n\n".join(context_chunks) if context_chunks else ""
    full_messages = []
    if ctx:
        full_messages.append(
            {
                "role": "system",
                "content": "Use the following document context when helpful:\n" + ctx,
            }
        )
    full_messages.extend(messages)

    url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": full_messages, "temperature": 0.2}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "") or "(empty response)"
    except Exception as e:
        return f"(DeepSeek error) {e}"


st.title("Fill DOCX via Chat")


if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": _greeting() + " If you have a .docx, upload it from the sidebar."})

with st.sidebar:
    st.subheader("Upload .docx")
    uploaded = st.file_uploader("Upload a .docx to fill", type=["docx"]) 
    if st.button("Clear chat"):
        st.session_state.messages = []
    if not _DOCX_FILL_ENABLED:
        st.error("python-docx not installed. Install: pip install python-docx")
    # Quick download of the current working DOCX
    if st.session_state.get("docx_work_path"):
        try:
            p = st.session_state.docx_work_path
            with open(p, "rb") as f:
                data = f.read()
            st.download_button(
                "Download current DOCX",
                data=data,
                file_name=os.path.basename(p),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        except Exception:
                    pass

# Process uploaded .docx
if uploaded is not None and _DOCX_FILL_ENABLED:
    if st.session_state.doc_name != uploaded.name:
        try:
            import tempfile
            tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getvalue())
            st.session_state.docx_work_path = tmp_path
            st.session_state.doc_name = uploaded.name
            st.session_state.blanks = [b.__dict__ for b in find_blanks_in_docx(tmp_path, ctx_words=20)]
            st.session_state.pending_blank = None
        except Exception as e:
            st.error(f"Failed to prepare DOCX: {e}")
            st.session_state.docx_work_path = ""
            st.session_state.blanks = []

if st.session_state.docx_work_path:
    st.caption(f"Loaded: {os.path.basename(st.session_state.docx_work_path)} ? Blanks remaining: {len(st.session_state.blanks)}")


if st.session_state.get("pending_confirmation"):
    pc = st.session_state.pending_confirmation
    b = pc.get("blank", {})
    label = b.get("label", "blank")
    extracted = pc.get("answer", "")
    expected = pc.get("expected", "text")
    display_extracted = extracted or pc.get("orig", "")
    st.warning(f"It looks like a '{expected}' is expected for '{label}', but the reply seems different.\n\nI extracted: {display_extracted}\n\nChoose to proceed or enter a new value.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Proceed with extracted value"):
            try:
            
                new_path = replace_one_blank(
                    st.session_state.docx_work_path,
                    paragraph_index=b["paragraph_index"],
                    start=b["start"],
                    end=b["end"],
                    replacement=extracted.strip(),
                )
                st.session_state.docx_work_path = new_path
                st.session_state.blanks = [x.__dict__ for x in find_blanks_in_docx(new_path, ctx_words=20)]
                try:
                    old_key = f"p{b['paragraph_index']}:{b['start']}-{b['end']}:{b['text']}"
                    st.session_state.blanks = [
                        bb for bb in st.session_state.blanks
                        if f"p{bb.get('paragraph_index')}:{bb.get('start')}-{bb.get('end')}:{bb.get('text')}" != old_key
                    ]
                except Exception:
                    pass
                st.session_state.pending_blank = None
                st.session_state.pending_confirmation = None
                remaining = len(st.session_state.blanks)
                confirm = f"Filled the blank with: {display_extracted}. Remaining blanks: {remaining}."
                st.session_state.messages.append({"role": "assistant", "content": confirm})
            except Exception as e:
                
                err = f"Error while filling (1): {e}"
                st.session_state.messages.append({"role": "assistant", "content": err})
    with c2:
        if st.button("Enter a different value"):
            st.session_state.pending_confirmation = None
            msg = f"Please provide a valid {expected} for '{label}'."
            st.session_state.messages.append({"role": "assistant", "content": msg})

# If fill mode is enabled and we have blanks, auto-ask the next question once
if st.session_state.docx_work_path:
    if st.session_state.pending_blank is None and st.session_state.blanks:
        b = st.session_state.blanks[0]
        st.session_state.pending_blank = b
        before = " ".join(b.get("before_words", [])).strip()
        after = " ".join(b.get("after_words", [])).strip()
        label = b.get("label", "blank")
        prefix = _greeting() if not st.session_state.messages else None
        try:
            import re as _re2
            expects_number = (
                b.get("kind") == "dollar_bracket" and _re2.fullmatch(r"^\$\[_+\]$", b.get("text", "")) is not None
            )
        except Exception:
            expects_number = b.get("kind") == "dollar_bracket"
        base_q = (
            f"Please provide a number for '{label}'." if expects_number else f"Please provide a value for '{label}'."
        )
        q = f"{prefix}\n\n{base_q}" if prefix else base_q
        st.session_state.messages.append({"role": "assistant", "content": q})
        # Replace with DeepSeek-generated question (greeting + plain-language explanation) if key is set
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if key:
            kind = b.get("kind", "unknown")
            ulen = b.get("underscore_len", 0)
            underscores_total = sum(1 for _bb in st.session_state.blanks if _bb.get("kind") == "underscore")
            system = (
                "You are a friendly assistant helping a user fill placeholders in a DOCX form. "
                "Use the provided context to infer in plain words what the blank represents, and ask ONE concise question. "
                "Begin with a short greeting only for the first blank. Output up to 2 lines: first the question, second starting with 'Why:' explaining simply."
            )
            
            user = (
                f"is_first: true\n"
                f"blank_index: 1 of {len(st.session_state.blanks)}\n"
                f"label: {label}\n"
                f"kind: {kind}\n"
                f"underscore_length: {ulen}\n"
                f"underscore_blanks_in_doc: {underscores_total}\n"
                f"before_context: {before}\n"
                f"after_context: {after}\n"
                
            )
            try:
                better_q = call_deepseek([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ], [], key)
                st.session_state.messages[-1]["content"] = better_q
            except Exception:
                    pass

# Chat display (no large fixed box)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type your answer for the blank")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # If there is a pending confirmation, treat this as a corrected value
    # If there is a pending confirmation, treat this as a corrected value
    if st.session_state.docx_work_path and st.session_state.pending_confirmation is not None:
        pc = st.session_state.pending_confirmation
        b = pc.get("blank", st.session_state.pending_blank)
        label = b.get("label", "blank")
        before = " ".join(b.get("before_words", [])).strip()
        after = " ".join(b.get("after_words", [])).strip()
        label = b.get("label", "blank")
        # If user typed an affirmation like "proceed", use the previously extracted value
        _skip = False
        if _is_yes(prompt):
            extracted = (pc.get("answer") or pc.get("orig") or "").strip()
            ok = True
            expected = pc.get("expected", "text")
        elif _is_no(prompt):
            # Ask for a different value explicitly
            st.session_state.pending_confirmation = None
            msg = f"Please provide a valid {pc.get('expected','text')} for '{label}'."
            st.session_state.messages.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.markdown(msg)
            # Stop further processing of this prompt
            _skip = True
            extracted = ""
            ok = False
            expected = pc.get("expected", "text")
        else:
            try:
                import re as _re2
                expected_hint = (
                    "number"
                    if (b.get("kind") == "dollar_bracket" and _re2.fullmatch(r"^\$\[_+\]$", b.get("text", "")) is not None)
                    else None
                )
            except Exception:
                expected_hint = "number" if b.get("kind") == "dollar_bracket" else None
            extracted, ok, expected = extract_and_validate(prompt, label, before, after, expected_hint)
        if not _skip and ok:
            try:
                new_path = replace_one_blank(
                    st.session_state.docx_work_path,
                    paragraph_index=b["paragraph_index"],
                    start=b["start"],
                    end=b["end"],
                    replacement=extracted,
                )
                st.session_state.docx_work_path = new_path
                st.session_state.blanks = [x.__dict__ for x in find_blanks_in_docx(new_path, ctx_words=20)]
                st.session_state.pending_blank = None
                st.session_state.pending_confirmation = None
                remaining = len(st.session_state.blanks)
                confirm = f"Filled the blank with: {extracted}. Remaining blanks: {remaining}."
                st.session_state.messages.append({"role": "assistant", "content": confirm})
                with st.chat_message("assistant"):
                    st.markdown(confirm)
            except Exception as e:
                err = f"Error while filling 2: {e}"
                st.session_state.messages.append({"role": "assistant", "content": err})
                with st.chat_message("assistant"):
                    st.markdown(err)
        elif not _skip:
            st.session_state.pending_confirmation = {
                "answer": extracted,
                "expected": expected,
                "blank": b,
                "orig": prompt,
            }
            warn = (
                f"It seems like a '{expected}' is expected. I extracted: {extracted}. ",
                "Would you like to proceed or enter a different value?"
            )
            st.session_state.messages.append({"role": "assistant", "content": "".join(warn)})
            with st.chat_message("assistant"):
                st.markdown("".join(warn))
    # If we are in fill mode and a blank is pending, treat the user's input as the answer
    elif st.session_state.docx_work_path and st.session_state.pending_blank is not None:
        b = st.session_state.pending_blank
        before = " ".join(b.get("before_words", [])).strip()
        after = " ".join(b.get("after_words", [])).strip()
        label = b.get("label", "blank")
        try:
            import re as _re2
            expected_hint = (
                "number"
                if (b.get("kind") == "dollar_bracket" and _re2.fullmatch(r"^\$\[_+\]$", b.get("text", "")) is not None)
                else None
            )
        except Exception:
            expected_hint = "number" if b.get("kind") == "dollar_bracket" else None
        extracted, ok, expected = extract_and_validate(prompt, label, before, after, expected_hint)
        try:
            if ok:
                new_path = replace_one_blank(
                st.session_state.docx_work_path,
                paragraph_index=b["paragraph_index"],
                start=b["start"],
                end=b["end"],
                replacement=extracted,
            )
            if ok:
                st.session_state.docx_work_path = new_path
                st.session_state.blanks = [x.__dict__ for x in find_blanks_in_docx(new_path, ctx_words=20)]
                st.session_state.pending_blank = None
                remaining = len(st.session_state.blanks)
                confirm = f"Filled the blank with: {extracted}. Remaining blanks: {remaining}."
                st.session_state.messages.append({"role": "assistant", "content": confirm})
                with st.chat_message("assistant"):
                    st.markdown(confirm)
            else:
                st.session_state.pending_confirmation = {
                    "answer": extracted,
                    "expected": expected,
                    "blank": b,
                    "orig": prompt,
                }
                msg = (
                    f"This field looks like a '{expected}'. I extracted: {extracted}.\n\n"
                    "Do you want to proceed with this value, or enter a different one?"
                )
                st.session_state.messages.append({"role": "assistant", "content": msg})
                with st.chat_message("assistant"):
                    st.markdown(msg)
            if ok:
                try:
                    with open(new_path, "rb") as f:
                        data = f.read()
                    
                    b64 = base64.b64encode(data).decode()
                    name = os.path.basename(new_path)
                    st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{name}">Download current DOCX</a>', unsafe_allow_html=True)
                except Exception:
                    pass
            if st.session_state.blanks and st.session_state.pending_confirmation is None:
                b = st.session_state.blanks[0]
                st.session_state.pending_blank = b
                before = " ".join(b.get("before_words", [])).strip()
                after = " ".join(b.get("after_words", [])).strip()
                label = b.get("label", "blank")
                try:
                    import re as _re2
                    expects_number = (
                        b.get("kind") == "dollar_bracket" and _re2.fullmatch(r"^\$\[_+\]$", b.get("text", "")) is not None
                    )
                except Exception:
                    expects_number = b.get("kind") == "dollar_bracket"
                base_q = (
                    f"Please provide a number for '{label}'." if expects_number else f"Please provide a value for '{label}'."
                )
                q = base_q
                st.session_state.messages.append({"role": "assistant", "content": q})
                with st.chat_message("assistant"):
                    st.markdown(q)
                key = os.environ.get("DEEPSEEK_API_KEY", "")
                if key:
                    kind = b.get("kind", "unknown")
                    ulen = b.get("underscore_len", 0)
                    underscores_total = sum(1 for _bb in st.session_state.blanks if _bb.get("kind") == "underscore")
                    system = (
                        "You are a friendly assistant helping a user fill placeholders in a DOCX form. "
                        "Use the provided context to infer in plain words what the blank represents, and ask ONE concise question. "
                        "Do not greet again. Output up to 2 lines: first the question, second starting with 'Why:' explaining simply."
                    )
                    user = (
                        f"is_first: false\n"
                        f"label: {label}\n"
                        f"kind: {kind}\n"
                        f"underscore_length: {ulen}\n"
                        f"underscore_blanks_in_doc: {underscores_total}\n"
                        f"before_context: {before}\n"
                        f"after_context: {after}\n"
                    )
                    try:
                        better_q = call_deepseek([
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ], [], key)
                        st.session_state.messages[-1]["content"] = better_q
                        # message already appended; rely on loop to render
                    except Exception:
                        print(" i a, here")
                        pass
        except Exception as e:
            print(e)
            err = f"Error while filling 3: {e}"
            st.session_state.messages.append({"role": "assistant", "content": err})
            with st.chat_message("assistant"):
                st.markdown(err)

    elif st.session_state.docx_work_path and not st.session_state.blanks:
        done = "All blanks are filled. You can download the document from above."
        st.session_state.messages.append({"role": "assistant", "content": done})
        with st.chat_message("assistant"):
            st.markdown(done)









