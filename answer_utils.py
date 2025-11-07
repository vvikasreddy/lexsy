import os
import re
from typing import Tuple, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


ORG_SUFFIXES = {"inc", "llc", "ltd", "corp", "co", "gmbh", "plc", "sa", "bv"}


def infer_expected_type(label: str, before: str, after: str) -> str:
    l = (label or "").lower()
    ctx = f"{before} {after}".lower()
    def has(*keys):
        return any(k in l or k in ctx for k in keys)

    # Highly specific types first
    if has("email"):
        return "email"
    if has("phone", "mobile", "contact number", "telephone"):
        return "phone"

    # Prefer person/name before organization/date/number to reduce misclassification
    if has("name", "full name", "firstname", "first name", "lastname", "last name", "investor", "applicant", "borrower", "beneficiary", "guarantor", "person", "contact", "representative", "signatory"):
        return "name"
    if has("company", "employer", "organization", "organisation", "org", "firm", "corp", "llc", "ltd", "plc", "gmbh"):
        return "organization"

    # Then date/number
    if has("date", "dob", "birth"):
        return "date"
    if has("age", "amount", "price", "quantity", "number", "count"):
        return "number"

    if has("address", "street", "city", "zip", "postcode"):
        return "address"
    return "text"


def _extract_regex(response: str, expected: str) -> str:
    s = response.strip()
    if expected == "email":
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", s)
        return m.group(0) if m else s
    if expected == "phone":
        m = re.search(r"(?:\+\d{1,3}[ \-]?)?(?:\(?\d{2,4}\)?[ \-]?){2,4}\d{2,4}", s)
        return m.group(0) if m else s
    if expected == "date":
        m = re.search(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b", s, re.I)
        return m.group(0) if m else s
    if expected == "number":
        m = re.search(r"[-+]?\d+(?:[\.,]\d+)?", s)
        return m.group(0) if m else s
    if expected == "organization":
        # Try capture token(s) after prepositions like 'at', 'with', 'for'
        m = re.search(r"\b(?:at|with|for|from)\s+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*)", s)
        if m:
            return m.group(1).strip()
        # Else take first capitalized phrase
        m = re.search(r"([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*)", s)
        return m.group(1).strip() if m else s
    if expected == "name":
        m = re.search(r"([A-Z][a-z]+\s+[A-Z][a-z]+)", s)
        return m.group(1) if m else s
    if expected == "address":
        m = re.search(r"\d+\s+[^,\n]+(?:,\s*[^,\n]+){0,2}", s)
        return m.group(0) if m else s
    return s


def _validate(ans: str, expected: str) -> Tuple[bool, str]:
    a = (ans or "").strip()
    if not a:
        return False, "empty"
    if expected == "email":
        ok = re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", a) is not None
        return ok, "email"
    if expected == "phone":
        digits = re.sub(r"\D", "", a)
        ok = 7 <= len(digits) <= 15
        return ok, "phone"
    if expected == "date":
        ok = bool(re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", a, re.I))
        return ok, "date"
    if expected == "number":
        ok = re.fullmatch(r"[-+]?\d+(?:[\.,]\d+)?", a) is not None
        return ok, "number"
    if expected == "organization":
        ok = len(a) >= 2 and not a.islower()
        return ok, "organization"
    if expected == "name":
        ok = len(a.split()) in (1, 2) and a[0].isupper()
        return ok, "name"
    if expected == "address":
        ok = any(ch.isdigit() for ch in a) and len(a) > 5
        return ok, "address"
    return True, "text"


def llm_extract_with_deepseek(response: str, label: str, before: str, after: str, expected: str, api_key: str) -> str:
    if not (api_key and requests):
        return _extract_regex(response, expected)
    sys = (
        "You extract a single value from a user's reply for a form field. "
        "Return ONLY the value, no explanations."
    )
    usr = (
        f"Field label: {label}\n"
        f"Expected type: {expected}\n"
        f"Context: …{before} [____] {after}…\n"
        f"User reply: {response}"
    )
    try:
        url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        payload = {"model": model, "messages": [{"role": "system", "content": sys}, {"role": "user", "content": usr}], "temperature": 0.1}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=30)  # type: ignore
        r.raise_for_status()
        j = r.json()
        val = j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not val:
            return _extract_regex(response, expected)
        lower = val.lower()
        if any(p in lower for p in ["unable to extract", "cannot extract", "not provided", "n/a", "none", "no value"]):
            return _extract_regex(response, expected)
        refined = _extract_regex(val, expected)
        if refined and refined != val:
            return refined
        ok, _ = _validate(val, expected)
        return val if ok else _extract_regex(response, expected)
    except Exception:
        return _extract_regex(response, expected)


def extract_and_validate(response: str, label: str, before: str, after: str, expected_override: Optional[str] = None) -> Tuple[str, bool, str]:
    expected = expected_override or infer_expected_type(label, before, after)
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        cand = llm_extract_with_deepseek(response, label, before, after, expected, key)
    else:
        cand = _extract_regex(response, expected)
    ok, kind = _validate(cand, expected)
    # Guard against placeholder-like outputs (e.g., "[____]")
    try:
        blank_like = re.fullmatch(r"\s*(\[\s*_{3,}\s*\]|_{3,})\s*", cand or "") is not None
    except Exception:
        blank_like = False
    if blank_like:
        ok = False

    # Normalize simple lowercase names: 1–2 alphabetic tokens, title-cased
    if expected == "name" and not ok:
        toks = re.findall(r"[A-Za-z][A-Za-z'-]*", response or cand or "")
        if toks:
            cand2 = " ".join(toks[:2]).title()
            ok2, _ = _validate(cand2, expected)
            if ok2:
                return cand2.strip(), ok2, expected

    return cand.strip(), ok, expected

