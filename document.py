# import os
# from typing import List

# import streamlit as st


# st.set_page_config(page_title="Document Viewer", page_icon="📄", layout="wide")
# st.title("Document Text Viewer")
# st.caption("View the text content of an uploaded .docx file from this session or upload one here.")


# # Try to enable DOCX parsing
# DOCX_ENABLED = True
# try:
#     from docx import Document  # type: ignore
# except Exception:
#     DOCX_ENABLED = False


# def extract_docx_text(path: str) -> str:
#     if not DOCX_ENABLED:
#         return ""
#     try:
#         doc = Document(path)
#         parts: List[str] = []
#         for p in doc.paragraphs:
#             txt = (p.text or "").strip()
#             if txt:
#                 parts.append(txt)
#         # Include simple table text too
#         for t in getattr(doc, "tables", []):
#             for row in t.rows:
#                 cells = [(c.text or "").strip() for c in row.cells]
#                 cells = [c for c in cells if c]
#                 if cells:
#                     parts.append(" | ".join(cells))
#         return "\n".join(parts)
#     except Exception:
#         return ""


# # Gather .docx uploads from session (if available)
# uploads = []
# if "uploads" in st.session_state and isinstance(st.session_state.uploads, list):
#     uploads = [u for u in st.session_state.uploads if str(u.get("save_path", "")).lower().endswith(".docx")]

# col1, col2 = st.columns([2, 1])
# with col1:
#     st.subheader("Pick from session uploads")
#     if not uploads:
#         st.info("No .docx found in this session's uploads.")
#         selected = None
#     else:
#         labels = [f"{i+1}. {os.path.basename(u['save_path'])}" for i, u in enumerate(uploads)]
#         idx = st.selectbox("Choose a document", options=list(range(len(uploads))), format_func=lambda i: labels[i] if labels else str(i), index=len(uploads)-1)
#         selected = uploads[idx]

# with col2:
#     st.subheader("Or upload here")
#     up = st.file_uploader("Upload a .docx", type=["docx"], key="local_docx")

# doc_text = ""
# source = ""

# if up is not None and DOCX_ENABLED:
#     # Read from in-memory upload on this page
#     import tempfile
#     tmp_path = os.path.join(tempfile.gettempdir(), up.name)
#     with open(tmp_path, "wb") as f:
#         f.write(up.getvalue())
#     doc_text = extract_docx_text(tmp_path)
#     source = f"(uploaded now) {tmp_path}"
# elif selected is not None:
#     # Read from previously uploaded file in this session
#     save_path = selected.get("save_path")
#     if save_path and os.path.exists(save_path) and DOCX_ENABLED:
#         doc_text = extract_docx_text(save_path)
#         source = f"(session upload) {save_path}"
#     elif save_path and not os.path.exists(save_path):
#         st.warning("Selected file no longer exists on disk (server may have restarted).")


# if not DOCX_ENABLED:
#     st.error("python-docx is not installed. Install it with: pip install python-docx")
# elif not doc_text:
#     st.info("Select a .docx from session uploads or upload one to see its text.")
# else:
#     st.success(f"Showing extracted text from: {source}")
#     st.text_area("Document text", value=doc_text, height=520)

