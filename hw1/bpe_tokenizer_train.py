import os
from collections import defaultdict
from typing import Dict, List, Tuple

import regex

GPT2_PRETOKENIZER_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _merge_token_sequence(
    token_seq: Tuple[bytes, ...],
    pair_to_merge: Tuple[bytes, bytes],
    merged_token: bytes,
) -> Tuple[bytes, ...]:
    """Merge all non-overlapping occurrences of pair_to_merge in token_seq."""
    merged: List[bytes] = []
    i = 0
    while i < len(token_seq):
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == pair_to_merge:
            merged.append(merged_token)
            i += 2
        else:
            merged.append(token_seq[i])
            i += 1
    return tuple(merged)


def _contains_pair(token_seq: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> bool:
    for i in range(len(token_seq) - 1):
        if (token_seq[i], token_seq[i + 1]) == pair:
            return True
    return False


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer vocabulary and merges."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")

    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    existing_vocab_values = set(vocab.values())
    for token in special_tokens:
        if len(vocab) >= vocab_size:
            break
        token_bytes = token.encode("utf-8")
        if token_bytes not in existing_vocab_values:
            vocab[next_token_id] = token_bytes
            existing_vocab_values.add(token_bytes)
            next_token_id += 1

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    segments = [text]
    if special_tokens:
        escaped_special_tokens = [regex.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True)]
        if escaped_special_tokens:
            segments = regex.split("|".join(escaped_special_tokens), text)

    token_frequency_table: defaultdict[Tuple[bytes, ...], int] = defaultdict(int)
    byte_to_singleton = [bytes([i]) for i in range(256)]

    for segment in segments:
        for pretoken in regex.findall(GPT2_PRETOKENIZER_PATTERN, segment):
            token_bytes = pretoken.encode("utf-8")
            token_frequency_table[tuple(byte_to_singleton[b] for b in token_bytes)] += 1

    pair_counts: defaultdict[Tuple[bytes, bytes], int] = defaultdict(int)
    for token_seq, frequency in token_frequency_table.items():
        for i in range(len(token_seq) - 1):
            pair_counts[(token_seq[i], token_seq[i + 1])] += frequency

    merges: List[Tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size and pair_counts:
        max_count = max(pair_counts.values())
        best_pair = max(pair for pair, count in pair_counts.items() if count == max_count)

        merged_token = best_pair[0] + best_pair[1]
        vocab[next_token_id] = merged_token
        next_token_id += 1
        merges.append(best_pair)

        affected_tokens = [
            (token_seq, frequency)
            for token_seq, frequency in token_frequency_table.items()
            if _contains_pair(token_seq, best_pair)
        ]

        for token_seq, frequency in affected_tokens:
            for i in range(len(token_seq) - 1):
                pair = (token_seq[i], token_seq[i + 1])
                pair_counts[pair] -= frequency
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]

            merged_sequence = _merge_token_sequence(token_seq, best_pair, merged_token)

            for i in range(len(merged_sequence) - 1):
                pair_counts[(merged_sequence[i], merged_sequence[i + 1])] += frequency

            del token_frequency_table[token_seq]
            token_frequency_table[merged_sequence] += frequency

    return vocab, merges
