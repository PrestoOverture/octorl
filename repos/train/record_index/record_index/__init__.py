"""Small document index with token positions, phrase search and reversible updates."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from collections.abc import Iterable
import re


@dataclass(frozen=True)
class Document:
    identifier: str
    title: str
    body: str
    tags: frozenset[str]


def tokenize(text: str) -> list[str]:
    """Normalize Unicode words; apostrophes separate tokens."""
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def positions(text: str) -> dict[str, tuple[int, ...]]:
    result: dict[str, list[int]] = {}
    for index, word in enumerate(tokenize(text)):
        result.setdefault(word, []).append(index)
    return {word: tuple(indices) for word, indices in result.items()}


def excerpt(text: str, query: str, width: int = 8) -> str:
    """Return up to width whitespace words centered near the first match."""
    if width <= 0:
        raise ValueError("positive excerpt width required")
    words = text.split()
    wanted = set(tokenize(query))
    match = next((index for index, word in enumerate(words) if wanted.intersection(tokenize(word))), 0)
    start = max(0, min(match - width // 2, len(words) - width))
    return " ".join(words[start:start + width])


class Index:
    """Document IDs are stable, ties in search rank sort by ID."""

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._postings: dict[str, dict[str, tuple[int, ...]]] = {}
        self._counts: dict[str, Counter[str]] = {}

    def __len__(self) -> int:
        return len(self._documents)

    def add(self, identifier: str, title: str, body: str,
            tags: Iterable[str] = ()) -> Document:
        if not identifier or identifier in self._documents:
            raise ValueError("new nonempty identifier required")
        normalized_tags = frozenset(tag.strip().casefold() for tag in tags if tag.strip())
        document = Document(identifier, title, body, normalized_tags)
        tokens = tokenize(title + " " + body)
        self._documents[identifier] = document
        self._counts[identifier] = Counter(tokens)
        for word, offsets in positions(title + " " + body).items():
            self._postings.setdefault(word, {})[identifier] = offsets
        return document

    def get(self, identifier: str) -> Document:
        return self._documents[identifier]

    def remove(self, identifier: str) -> Document:
        document = self._documents.pop(identifier)
        for word in self._counts.pop(identifier):
            del self._postings[word][identifier]
            if not self._postings[word]:
                del self._postings[word]
        return document

    def replace(self, identifier: str, title: str, body: str,
                tags: Iterable[str] | None = None) -> Document:
        previous = self.get(identifier)
        selected_tags = previous.tags if tags is None else tuple(tags)
        self.remove(identifier)
        return self.add(identifier, title, body, selected_tags)

    def vocabulary(self, prefix: str = "") -> tuple[str, ...]:
        prefix = prefix.casefold()
        return tuple(sorted(word for word in self._postings if word.startswith(prefix)))

    def frequency(self, word: str, identifier: str | None = None) -> int:
        normalized = tokenize(word)
        if len(normalized) != 1:
            raise ValueError("one word required")
        word = normalized[0]
        if identifier is not None:
            return self._counts[identifier][word]
        return sum(len(offsets) for offsets in self._postings.get(word, {}).values())

    def search(self, query: str, mode: str = "all", tag: str | None = None,
               limit: int | None = None) -> list[str]:
        if mode not in {"all", "any"}:
            raise ValueError("mode must be all or any")
        if limit is not None and limit < 0:
            raise ValueError("negative limit")
        words = set(tokenize(query))
        if not words:
            return []
        matches = [set(self._postings.get(word, {})) for word in words]
        identifiers = set.intersection(*matches) if mode == "all" else set.union(*matches)
        if tag is not None:
            identifiers = {identifier for identifier in identifiers
                           if tag.casefold().strip() in self._documents[identifier].tags}
        ranked = sorted(identifiers, key=lambda identifier: (
            -sum(self._counts[identifier][word] for word in words), identifier))
        return ranked if limit is None else ranked[:limit]

    def phrase(self, query: str) -> list[str]:
        words = tokenize(query)
        if not words:
            return []
        result = []
        for identifier in self.search(query):
            starts = self._postings[words[0]][identifier]
            if any(all(start + offset in self._postings[word][identifier]
                       for offset, word in enumerate(words)) for start in starts):
                result.append(identifier)
        return result

    def tag_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for document in self._documents.values():
            counts.update(document.tags)
        return dict(sorted(counts.items()))

    def related(self, identifier: str, limit: int = 5) -> list[str]:
        if limit < 0:
            raise ValueError("negative limit")
        words = set(self._counts[identifier])
        scores = []
        for other, counts in self._counts.items():
            if other != identifier:
                overlap = len(words.intersection(counts))
                if overlap > 0:
                    scores.append((-overlap, other))
        return [other for _, other in sorted(scores)[:limit]]

    def export(self) -> list[dict[str, object]]:
        return [dict(identifier=document.identifier, title=document.title,
                     body=document.body, tags=sorted(document.tags))
                for document in sorted(self._documents.values(), key=lambda item: item.identifier)]

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, object]]) -> Index:
        index = cls()
        for row in rows:
            identifier, title, body = row["identifier"], row["title"], row["body"]
            tags = row.get("tags", [])
            if not all(isinstance(value, str) for value in (identifier, title, body)):
                raise ValueError("document fields must be strings")
            if not isinstance(tags, (list, tuple)) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError("tags must be a string sequence")
            index.add(identifier, title, body, tags)
        return index

    def audit(self) -> bool:
        rebuilt = Index.from_rows(self.export())
        return rebuilt._postings == self._postings and rebuilt._counts == self._counts


def parse_query(query: str) -> tuple[list[str], list[str]]:
    """Split required and minus-prefixed excluded words."""
    include, exclude = [], []
    for part in query.split():
        if part.startswith("-"):
            exclude.extend(tokenize(part[1:]))
        else:
            include.extend(tokenize(part))
    return include, exclude


def filtered_search(index: Index, query: str) -> list[str]:
    include, exclude = parse_query(query)
    candidates = index.search(" ".join(include))
    blocked = set(index.search(" ".join(exclude), mode="any"))
    return [identifier for identifier in candidates if identifier not in blocked]


def paginate(items: Iterable[str], page: int, size: int) -> tuple[list[str], bool]:
    """One-based pagination with a next-page flag."""
    if page < 1 or size < 1:
        raise ValueError("positive page and size required")
    values = list(items)
    start = (page - 1) * size
    return values[start:start + size], start + size < len(values)


def concordance(index: Index, word: str) -> list[tuple[str, int]]:
    """All documents containing a word, sorted by decreasing frequency."""
    return [(identifier, index.frequency(word, identifier)) for identifier in index.search(word)]
