# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse token ids across chat prompts that share rendered segments.

A multi-turn chat request re-sends the whole conversation, so encoding its
rendered prompt costs time linear in the history on every turn. Fast
tokenizers split the text on added tokens before normalization and
pre-tokenization, then encode each piece on its own. The rendered prompt is
cut before each such token, every segment is encoded independently, and the
ids of segments seen before are taken from an LRU cache. The concatenated ids
equal a full encode when:

- the cut tokens are not normalized and strip no neighboring whitespace;
- they match only whole: no other added token contains them, which the
  leftmost-longest scan below enforces;
- each cut segment starts with its token, so its text is never at offset 0
  (where some pre-tokenizers prepend a word marker), and that token starts
  with a non-word character (so no `single_word` token at the end of the
  previous segment depends on what follows).
"""

import threading
from array import array
from collections import OrderedDict

import regex as re

from vllm.logger import init_logger
from vllm.tokenizers.hf import HfTokenizer

logger = init_logger(__name__)


class ChatEncodeCache:
    def __init__(self, tokenizer: HfTokenizer, max_bytes: int) -> None:
        self.tokenizer = tokenizer
        self.max_bytes = max_bytes

        # Every unnormalized added token takes part in the tokenizer's
        # leftmost-longest match; only some are safe to cut before.
        matched: list[str] = []
        self._cut: set[str] = set()
        for token in tokenizer.added_tokens_decoder.values():
            if token.normalized or not token.content:
                continue
            matched.append(token.content)
            first = token.content[0]
            if not (
                token.lstrip
                or token.rstrip
                or token.single_word
                or first.isalnum()
                or first == "_"
            ):
                self._cut.add(token.content)
        matched.sort(key=len, reverse=True)
        self._pattern = re.compile("|".join(map(re.escape, matched)))

        self._lock = threading.Lock()
        self._segments: OrderedDict[str, array] = OrderedDict()
        self._num_bytes = 0

    @classmethod
    def maybe_create(
        cls, tokenizer: HfTokenizer | None, max_bytes: int
    ) -> "ChatEncodeCache | None":
        if tokenizer is None or not getattr(tokenizer, "is_fast", False):
            reason = "it needs a fast tokenizer"
        elif getattr(tokenizer, "split_special_tokens", False):
            reason = "the tokenizer does not split on special tokens"
        else:
            cache = cls(tokenizer, max_bytes)
            if cache._cut:
                return cache
            reason = "the tokenizer has no added token to cut at"
        logger.warning("Incremental chat tokenization is disabled: %s.", reason)
        return None

    def split(self, text: str) -> list[str]:
        segments = []
        start = 0
        for match in self._pattern.finditer(text):
            if match.start() > start and match.group() in self._cut:
                segments.append(text[start : match.start()])
                start = match.start()
        segments.append(text[start:])
        return segments

    def encode(self, text: str) -> list[int]:
        """Return `tokenizer.encode(text, add_special_tokens=False)`."""
        segments = self.split(text)
        found: dict[str, array] = {}
        with self._lock:
            for segment in segments:
                ids = self._segments.get(segment)
                if ids is not None:
                    self._segments.move_to_end(segment)
                    found[segment] = ids
        missing = list(dict.fromkeys(s for s in segments if s not in found))
        if missing:
            encoded = self.tokenizer(missing, add_special_tokens=False)["input_ids"]
            new = {s: array("i", ids) for s, ids in zip(missing, encoded)}
            found.update(new)
            self._insert(new)

        token_ids: list[int] = []
        for segment in segments:
            token_ids.extend(found[segment])
        return token_ids

    def _insert(self, new: dict[str, array]) -> None:
        with self._lock:
            for segment, ids in new.items():
                if segment in self._segments:
                    continue
                self._segments[segment] = ids
                self._num_bytes += _num_bytes(segment, ids)
            while self._num_bytes > self.max_bytes:
                segment, ids = self._segments.popitem(last=False)
                self._num_bytes -= _num_bytes(segment, ids)


def _num_bytes(segment: str, ids: array) -> int:
    # Approximate: one byte per character plus the stored ids.
    return len(segment) + ids.itemsize * len(ids)
