import re
from difflib import get_close_matches
from typing import Iterable, Optional


GENERIC_SOURCE_WORDS = {
    "document",
    "doc",
    "file",
    "pdf",
    "txt",
    "docx",
    "guide",
    "program",
    "policy",
    "standard",
    "guidance",
}

TOKEN_EXPANSIONS = {
    "osha": ["occupational", "safety", "health", "administration"],
    "dol": ["department", "labor"],
    "dept": ["department"],
}


def normalize_text(value: str) -> str:
    if not value:
        return ""

    value = value.strip().lower()
    value = re.sub(r"\.[a-z0-9]+$", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def _tokenize(value: str) -> list[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", (value or "").lower())
    tokens: list[str] = []

    for token in raw_tokens:
        if len(token) > 1:
            tokens.append(token)

        expansions = TOKEN_EXPANSIONS.get(token, [])
        tokens.extend(expansions)

    return tokens


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token
        for token in _tokenize(value)
        if token not in GENERIC_SOURCE_WORDS and len(token) > 2
    }


def build_source_aliases(source: str, title: Optional[str] = None) -> list[str]:
    aliases = set()

    if source:
        aliases.add(source.strip())
        aliases.add(normalize_text(source))
        aliases.add(source.rsplit(".", 1)[0].strip())

    if title:
        aliases.add(title.strip())
        aliases.add(normalize_text(title))

    source_tokens = sorted(_meaningful_tokens(source or ""))
    if source_tokens:
        aliases.add(" ".join(source_tokens))

    return sorted(a for a in aliases if a)


def resolve_target_source(target: str, known_sources: Iterable[str]) -> Optional[str]:
    known_sources = list(known_sources or [])
    if not target or not known_sources:
        return None

    target_norm = normalize_text(target)
    target_tokens = _meaningful_tokens(target)

    normalized_map = {}
    for source in known_sources:
        normalized_map[normalize_text(source)] = source

    if target_norm in normalized_map:
        return normalized_map[target_norm]

    for norm_source, original_source in normalized_map.items():
        if target_norm in norm_source or norm_source in target_norm:
            return original_source

    best_source = None
    best_score = 0.0

    for source in known_sources:
        source_tokens = _meaningful_tokens(source)
        if not source_tokens or not target_tokens:
            continue

        overlap = target_tokens & source_tokens
        if not overlap:
            continue

        coverage = len(overlap) / max(len(target_tokens), 1)
        source_coverage = len(overlap) / max(len(source_tokens), 1)
        score = (coverage * 0.75) + (source_coverage * 0.25)

        if score > best_score:
            best_score = score
            best_source = source

    if best_source and best_score >= 0.45:
        return best_source

    matches = get_close_matches(
        target_norm,
        list(normalized_map.keys()),
        n=1,
        cutoff=0.65,
    )
    if matches:
        return normalized_map[matches[0]]

    return None