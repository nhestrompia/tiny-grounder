"""Hard safety and privacy rules for macOS trajectory collection.

The app split policy is data (`configs/apps_policy.yaml`), but these rules are code on purpose: a config edit
must never be able to weaken the click denylist, typing boundary, sign-in refusal or privacy scrubber.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

# Hard global click denylist mandated for the collector. Matching is case-insensitive substring matching.
HARD_CLICK_DENYLIST: tuple[str, ...] = tuple(sorted({
    "buy", "purchase", "subscribe", "pay", "checkout", "order", "export", "share", "publish", "upload",
    "send", "delete", "remove", "erase", "trash", "empty", "format", "partition", "commit", "push", "merge",
    "kill", "terminate", "stop container", "force quit", "sign out", "log out", "sign in", "install",
    "update now", "restart", "shut down", "reset", "revoke", "unlink",
    # Account, library and permission changes (review 2026-11-26).
    "follow", "like", "save", "download", "accept", "agree", "allow", "enable", "turn on", "turn off",
    "clear", "add to library", "add to playlist", "rate this", "write a review",
}))

# Whole-label denials: short labels that are too common as substrings ("Get Info", "Target") but mean
# install/consent on their own (App Store "Get", permission prompts).
HARD_EXACT_DENYLIST: frozenset[str] = frozenset({"get", "ok", "on", "off", "yes", "continue", "upgrade"})

FIXED_TYPE_STRINGS: tuple[str, ...] = (
    "search test",
    "apple",
    "weather",
    "music",
    "document notes",
)

FORBIDDEN_VIEW_WORDS = frozenset({"history", "account", "profile"})
CREDENTIAL_WORDS = frozenset({"password", "passkey", "verification code", "one-time code", "otp"})
PAYMENT_WORDS = frozenset({"credit card", "debit card", "card number", "cvv", "payment", "billing"})

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
# Phone numbers in the formats people write them: "+" international, North American 3-3-4 and the
# 0xxx xxx xx xx trunk style. Bare digit runs are not phones here: prices ("50.656,19"), model numbers, years, RGB
# triples and versions are ordinary UI text.
_PHONE_RE = re.compile(
    r"(?<![\w.,+])(?:"
    r"\+\d{1,3}[ .-]?(?:\(\d{1,4}\)[ .-]?)?\d{2,4}(?:[ .-]?\d{2,4}){1,4}"
    r"|\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}"
    r"|0\d{2,4}[ -]\d{3}[ -]\d{2}[ -]?\d{2}"
    r")(?![\w,])"
)
# URLs carrying query strings, fragments with tokens, or obvious token/key parameters.
_URL_WITH_SECRET_RE = re.compile(
    r"\bhttps?://[^\s<>'\"]+(?:\?[^\s<>'\"]+|#[^\s<>'\"]*(?:token|access|auth|sig|key)[^\s<>'\"]*)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"\b(history|account|profile)s?\b", re.IGNORECASE)


def denied_click_term(label: str, extra_terms: Iterable[str] = ()) -> str:
    """The denylist term found in `label`, or ''. Case-insensitive substring matching."""
    folded = label.casefold()
    if folded.strip() in HARD_EXACT_DENYLIST:
        return folded.strip()
    for term in (*HARD_CLICK_DENYLIST, *extra_terms):
        if term.casefold() in folded:
            return term
    return ""


def is_search_element(role: str, tag: str, label: str) -> bool:
    """True only for accessibility search fields, not arbitrary text boxes."""
    haystack = " ".join((role, tag, label)).casefold()
    return role == "searchbox" or "searchfield" in tag.casefold() or re.search(r"\bsearch\b", haystack) is not None


def typing_allowed(mode: str, role: str, tag: str, label: str, throwaway_document: bool) -> bool:
    """Typing boundary: search fields, or the throwaway document of a new empty project/note/board only."""
    if mode == "never":
        return False
    if mode == "search":
        return is_search_element(role, tag, label)
    if mode == "throwaway":
        return throwaway_document and role == "textbox"
    return False


def fixed_type_text(seed: int) -> str:
    """One fixed harmless input string; collectors never type user-derived or free-form text."""
    return FIXED_TYPE_STRINGS[seed % len(FIXED_TYPE_STRINGS)]


def contains_forbidden_view(text: str) -> str:
    """The forbidden view word in `text` (history/account/profile), or ''."""
    match = _WORD_RE.search(text)
    return match.group(1).casefold() if match else ""


def signin_or_payment_reason(raw: dict[str, Any]) -> str:
    """Why this raw dump is a sign-in/password/payment screen, or '' when it is not.

    A mere "Sign in" button is not enough (many browsable apps show one); a secure field, credential text in
    an editable field, or payment-card terms in editable controls is. Clicking those buttons is separately denied.
    """
    elements = raw.get("elements") or []
    secure = any((e.get("tag") or "") == "AXSecureTextField" for e in elements if isinstance(e, dict))
    if secure:
        return "secure password field"
    editable_text = []
    for e in elements:
        if not isinstance(e, dict):
            continue
        if e.get("role") in {"textbox", "searchbox", "combobox"} or (e.get("tag") or "") in {"AXTextField", "AXTextArea"}:
            editable_text.append(" ".join(str(e.get(k) or "") for k in ("text", "context", "visible_text", "nonvisual_text")))
    joined = " ".join(editable_text).casefold()
    title = str(raw.get("title") or "").casefold()
    if any(word in joined for word in CREDENTIAL_WORDS):
        return "credential entry"
    if editable_text and re.search(r"\b(sign in|log in|login|create account)\b", title):
        return "sign-in screen"
    for e in elements:
        if not isinstance(e, dict):
            continue
        tag = str(e.get("tag") or "")
        role = str(e.get("role") or "")
        if role == "searchbox" or "searchfield" in tag.casefold():
            continue
        if role in {"textbox", "combobox"} or tag in {"AXTextField", "AXTextArea"}:
            evidence = " ".join(str(e.get(k) or "") for k in ("text", "context", "visible_text", "nonvisual_text")).casefold()
            if re.search(r"\b(sign in|log in|login|email|username|user name)\b", evidence):
                return "sign-in screen"
    if sum(1 for word in PAYMENT_WORDS if word in joined) >= 2:
        return "payment entry"
    return ""


@dataclass
class Scrubber:
    """Deterministic privacy scrubber with aggregate replacement counts."""

    full_name: str
    username: str
    counts: Counter[str] = field(default_factory=Counter)

    @classmethod
    def from_system(cls) -> "Scrubber":
        def command(args: list[str]) -> str:
            try:
                return subprocess.run(args, capture_output=True, text=True, timeout=2).stdout.strip()
            except Exception:
                return ""
        return cls(command(["id", "-F"]), command(["whoami"]))

    def scrub_text(self, text: str) -> str:
        """Replace emails, phones, token/query URLs and local user identifiers; count every replacement."""
        if not text:
            return text

        def replace(pattern: re.Pattern[str], value: str, key: str, source: str) -> str:
            matches = pattern.findall(source)
            if matches:
                self.counts[key] += len(matches)
            return pattern.sub(value, source)

        out = replace(_EMAIL_RE, "<email>", "emails", text)
        out = replace(_PHONE_RE, "<phone>", "phone_numbers", out)
        out = replace(_URL_WITH_SECRET_RE, "<url-with-query>", "urls_with_tokens_or_queries", out)
        for source, replacement, key in (
            (self.full_name, "<macos-full-name>", "macos_full_names"),
            (self.username, "<macos-username>", "macos_usernames"),
        ):
            if source:
                n = out.count(source)
                if n:
                    self.counts[key] += n
                    out = out.replace(source, replacement)
        return out

    def name_tokens(self) -> list[str]:
        """Folded, accent-free parts of the account's full name and username (4+ characters), for OCR matching."""
        tokens = re.split(r"[\s._-]+", f"{self.full_name} {self.username}")
        return sorted({fold_accents(t) for t in tokens if len(t) >= 4})

    def contains_private(self, text: str) -> bool:
        """Whether text shows an email, phone, token URL or the account's name/username (OCR-tolerant)."""
        if not text:
            return False
        if _EMAIL_RE.search(text) or _PHONE_RE.search(text) or _URL_WITH_SECRET_RE.search(text):
            return True
        folded = fold_accents(text)
        return any(token in folded for token in self.name_tokens())

    def scrub_raw_dump(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of an axdump JSON object with every user-visible string field scrubbed."""
        scrubbed = dict(raw)
        scrubbed["title"] = self.scrub_text(str(raw.get("title") or ""))
        elements = []
        for element in raw.get("elements") or []:
            if not isinstance(element, dict):
                continue
            row = dict(element)
            for key in ("text", "context", "visible_text", "nonvisual_text", "href"):
                if key in row:
                    row[key] = self.scrub_text(str(row.get(key) or ""))
            elements.append(row)
        scrubbed["elements"] = elements
        return scrubbed


PRIVATE_PLACEHOLDERS = ("<email>", "<phone>", "<url-with-query>", "<macos-full-name>", "<macos-username>")


def fold_accents(text: str) -> str:
    """Casefolded text without diacritics, so OCR output like "Ozcelik" matches "Özçelik"."""
    import unicodedata
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).replace("ı", "i")


def scrubbed_element_boxes(raw: dict[str, Any]) -> list[list[float]]:
    """Screen boxes of elements whose text the scrubber replaced (so the screenshot shows something private)."""
    boxes = []
    for element in raw.get("elements") or []:
        if not isinstance(element, dict):
            continue
        text = " ".join(str(element.get(k) or "") for k in ("text", "visible_text", "nonvisual_text"))
        box = element.get("abs_box")
        if box and any(p in text for p in PRIVATE_PLACEHOLDERS):
            boxes.append([float(v) for v in box])
    return boxes


_FILE_DIALOG_TITLES = {"open", "save", "save as", "export", "import", "choose", "choose a file", "select"}
_FILE_DIALOG_HINTS = {"where:", "icon view", "list view", "column view", "new folder", "show options"}


def file_dialog_reason(raw: dict[str, Any]) -> str:
    """Why a dump is a file open/save panel: it lists the user's own files and folders, so it is never saved."""
    labels = {str(e.get("text") or "").strip().casefold() for e in raw.get("elements") or [] if isinstance(e, dict)}
    title = str(raw.get("title") or "").strip().casefold()
    has_cancel = "cancel" in labels
    has_commit = bool(labels & {"open", "save", "choose", "export", "import", "select"})
    if has_cancel and has_commit and (title in _FILE_DIALOG_TITLES or labels & _FILE_DIALOG_HINTS):
        return "file dialog"
    return ""


def private_view_reason(raw: dict[str, Any], extra_terms: Iterable[str] = ()) -> str:
    """Why a dump belongs to a history/account/profile (or app-denied) view and must never be saved."""
    title = str(raw.get("title") or "")
    found = contains_forbidden_view(title)
    if found:
        return f"{found} view"
    terms = {term.casefold() for term in extra_terms}
    if any(term and term in title.casefold() for term in terms):
        return "policy-denied view"
    # Window titles are sometimes empty in macOS apps. Treat repeated forbidden words in element contexts as view
    # evidence, but do not skip a window merely because one link happens to contain the word.
    contexts = " ".join(str(e.get("context") or "") for e in raw.get("elements") or [] if isinstance(e, dict))
    matches = _WORD_RE.findall(contexts)
    if len(matches) >= 3:
        return f"{matches[0].casefold()} view"
    for term in terms:
        if term and contexts.casefold().count(term) >= 3:
            return "policy-denied view"
    return ""
