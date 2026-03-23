import re
from difflib import get_close_matches
from typing import Iterable, Optional


def normalize_text(value: str) -> str:
    if not value:
        return ""

    value = value.strip().lower()
    value = re.sub(r"\.[a-z0-9]+$", "", value)   # remove extension
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def build_source_aliases(source: str, title: Optional[str] = None) -> list[str]:
    aliases = set()

    if source:
        aliases.add(source.strip())
        aliases.add(normalize_text(source))

    if title:
        aliases.add(title.strip())
        aliases.add(normalize_text(title))

    return sorted(a for a in aliases if a)


def resolve_target_source(target: str, known_sources: Iterable[str]) -> Optional[str]:
    known_sources = [s for s in known_sources if s]
    if not target or not known_sources:
        return None

    target_norm = normalize_text(target)

    normalized_map = {}
    for source in known_sources:
        normalized_map[normalize_text(source)] = source

    # exact normalized match
    if target_norm in normalized_map:
        return normalized_map[target_norm]

    # partial semantic-ish match
    for norm_source, original_source in normalized_map.items():
        if target_norm in norm_source or norm_source in target_norm:
            return original_source

    # fuzzy fallback
    matches = get_close_matches(
        target_norm,
        list(normalized_map.keys()),
        n=1,
        cutoff=0.72,
    )
    if matches:
        return normalized_map[matches[0]]

    return None