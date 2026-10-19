"""CPU-only R5 intent classification and indexed catalog entity resolution.

The encoder is frozen. A small linear classifier is fitted to versioned examples
once per process. No generation, network inference, or playlist mutation occurs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import json
import os
import re
import unicodedata

import numpy as np


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


# Grammatical words must not become entities merely because a catalog contains
# a song called "Who" or "When". Quoted names can override this stop list.
GRAMMAR = set("a an the who what which when where how is was are were do does did made by from on of in for to me my you your we our us this that it they their same song songs track tracks music artist artists album release released date year genre style tags other else more out come came sings sang sung said performs performed recorded tell show list name please anything something band singer record kind can could would should have has had".split())


@dataclass(frozen=True)
class Mention:
    name: str
    start: int
    end: int
    ids: frozenset[str]


class CatalogEntities:
    def __init__(self, catalog):
        self.index = {
            "title": defaultdict(set),
            "artist": defaultdict(set),
            "album": defaultdict(set),
        }
        self.full_names = {kind: defaultdict(set) for kind in self.index}
        for track_id, row in catalog.metadata_dict.items():
            for kind, field in (("title", "track_name"), ("artist", "artist_name"), ("album", "album_name")):
                values = row.get(field) or []
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    name = normalize(str(value))
                    if name:
                        self.index[kind][name].add(track_id)
                        self.full_names[kind][name].add(track_id)
                        # A distinctive word is a useful album shorthand
                        # ("Issa" for "Issa Album"). Skip generic words.
                        if kind == "album":
                            for token in name.split():
                                if len(token) >= 4 and token not in GRAMMAR:
                                    self.index[kind][token].add(track_id)
        self.max_words = {kind: max((len(n.split()) for n in names), default=1)
                          for kind, names in self.index.items()}
        self.compact_names = {kind: defaultdict(set) for kind in self.index}
        for kind, names in self.full_names.items():
            for name in names:
                self.compact_names[kind][name.replace(" ", "")].add(name)

    def match_name(self, text: str, kind: str, *, fuzzy: bool = True) -> list[Mention]:
        """Resolve a complete entity slot, without matching question words.

        Full names exclude the album shorthand aliases used by free-text
        mentions. Whitespace-only differences are safe here ("savagemode").
        Numbered releases must retain their suffix even during fuzzy lookup.
        """
        query = normalize(text)
        names = self.full_names[kind]
        compact = query.replace(" ", "")
        hits = sorted(self.compact_names[kind].get(compact, ()))
        if not hits and fuzzy and len(compact) >= 5:
            try:
                from rapidfuzz import process, fuzz
            except ImportError:
                return []
            suffix = re.search(r"(?:^|\s)(\d+|ii|iii|iv|v|vi|vii|viii|ix|x)$", query)
            suffix = suffix.group(1) if suffix else None
            scored = process.extract(query, names.keys(), scorer=fuzz.ratio,
                                     score_cutoff=85, limit=5)
            eligible = []
            for name, score, _ in scored:
                candidate_suffix = re.search(r"(?:^|\s)(\d+|ii|iii|iv|v|vi|vii|viii|ix|x)$", name)
                candidate_suffix = candidate_suffix.group(1) if candidate_suffix else None
                if candidate_suffix == suffix:
                    eligible.append((name, score))
            if eligible:
                hits = sorted(name for name, score in eligible if score >= eligible[0][1] - 3)
        return [Mention(name, 0, len(query.split()), frozenset(names[name])) for name in hits]

    def mentions(self, text: str, kind: str, fuzzy: bool = False) -> list[Mention]:
        words = normalize(text).split()
        quoted = {normalize(s) for s in re.findall(r'["“]([^"”]+)["”]', text)}
        candidates = []
        names = self.index[kind]
        album_context = bool(re.search(r"\b(album|record|tracklist|on|in|from)\b", text, re.I))
        for start in range(len(words)):
            for end in range(start + 1, min(len(words), start + self.max_words[kind]) + 1):
                name = " ".join(words[start:end])
                if name not in quoted and all(w in GRAMMAR for w in words[start:end]):
                    continue
                if kind == "album" and name not in self.full_names[kind] and not album_context:
                    continue
                if name in names:
                    candidates.append(Mention(name, start, end, frozenset(names[name])))
        if not candidates and fuzzy:
            try:
                from rapidfuzz import process, fuzz
            except ImportError:
                # Keep exact and context-based lookup usable when optional
                # typo-tolerant matching has not been installed yet.
                return []
            # Only compare plausible entity spans, not whole questions. A small
            # edit is allowed; close competing names are retained for clarification.
            for start in range(len(words)):
                if words[start] in GRAMMAR:
                    continue
                for end in range(start + 1, min(len(words), start + min(self.max_words[kind], 8)) + 1):
                    if words[end - 1] in GRAMMAR:
                        continue
                    query = " ".join(words[start:end])
                    if len(query) < 4:
                        continue
                    # Short title spans are too ambiguous for fuzzy expansion:
                    # e.g. "Them" should not silently become "Theme". Exact
                    # title mentions are still matched above.
                    if kind == "title" and len(query.replace(" ", "")) < 5:
                        continue
                    hits = process.extract(query, names.keys(), scorer=fuzz.ratio,
                                           score_cutoff=85, limit=2)
                    for name, score, _ in hits:
                        if len(name) >= 4 and score >= hits[0][1] - 3:
                            candidates.append(Mention(name, start, end, frozenset(names[name])))
        # Longest spans first; preserve ties (e.g. competing fuzzy names).
        selected = []
        for mention in sorted(candidates, key=lambda m: (-(m.end - m.start), m.start)):
            if not any(mention.start < m.end and m.start < mention.end
                       and (mention.start, mention.end) != (m.start, m.end) for m in selected):
                selected.append(mention)
        return selected

    def mask(self, text: str, title_mentions=None, artist_mentions=None, album_mentions=None) -> str:
        words = normalize(text).split()
        artist_mentions = artist_mentions if artist_mentions is not None else self.mentions(text, "artist")
        title_mentions = title_mentions if title_mentions is not None else self.mentions(text, "title")
        album_mentions = album_mentions if album_mentions is not None else self.mentions(text, "album")
        spans = [(m, "this artist") for m in artist_mentions]
        spans += [(m, "this song") for m in title_mentions]
        spans += [(m, "this album") for m in album_mentions]
        replacements = {}
        occupied = set()
        for m, replacement in sorted(spans, key=lambda pair: -(pair[0].end - pair[0].start)):
            replacement_start = m.start
            if replacement.startswith("this ") and replacement_start > 0 and words[replacement_start - 1] in {"a", "an", "the"}:
                replacement_start -= 1
            positions = set(range(replacement_start, m.end))
            if not positions & occupied:
                replacements[replacement_start] = replacement
                occupied.update(positions)
        return " ".join(replacements.get(i, word) for i, word in enumerate(words)
                        if i not in occupied or i in replacements)


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float
    margin: float
    accepted: bool


class R5IntentRouter:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        root = Path(__file__).resolve().parents[1]
        model = os.environ.get("MUSICCRS_NLU_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = SentenceTransformer(
            model, device="cpu", cache_folder=str(root / "cache/nlu/models"),
            local_files_only=os.environ.get("MUSICCRS_NLU_LOCAL_ONLY") == "1",
        )
        self.encoder.max_seq_length = 128
        examples = json.loads((Path(__file__).parent / "nlu_data/r5_train.json").read_text())
        texts, labels = [], []
        for intent, phrases in examples.items():
            texts.extend(phrases)
            labels.extend([intent] * len(phrases))
        vectors = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.classifier = LogisticRegression(C=20, max_iter=1000, class_weight="balanced", random_state=0)
        self.classifier.fit(vectors, labels)
        self._lock = Lock()

    def predict(self, text: str) -> IntentPrediction:
        if len(text) > 1000:
            return IntentPrediction("other", 0, 0, False)
        with self._lock:
            vector = self.encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)
            scores = self.classifier.predict_proba(vector)[0]
        ranked = np.argsort(scores)
        best, second = ranked[-1], ranked[-2]
        confidence = float(scores[best])
        margin = confidence - float(scores[second])
        intent = str(self.classifier.classes_[best])
        return IntentPrediction(intent, confidence, margin,
                                intent != "other" and confidence >= 0.55 and margin >= 0.18)


_router = None
_router_lock = Lock()


def get_r5_router() -> R5IntentRouter:
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                _router = R5IntentRouter()
    return _router
