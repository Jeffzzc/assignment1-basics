import os
from collections import Counter
from collections import defaultdict
import regex as re
import multiprocessing
from pretokenization_example import find_chunk_boundaries

import time


# 预编译正则
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pre_tokenization(chunk, special_tokens) -> list[str]:
    if not special_tokens:
        return [m.groups() for m in PAT.finditer(chunk)]
    
    special_set = set(special_tokens)
    sorted_special = sorted(special_tokens, key=len, reverse=True)
    special_pat = "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
    parts = re.split(special_pat, chunk)

    result = []
    for part in parts:
        if not part: continue
        if part in special_set:
            result.append(part)
        else:
            matches = PAT.finditer(part)
            for m in matches:
                result.append(m.group())
    return result

def process_chunk(input_path, start, end, special_tokens) -> dict[tuple[int, ...], int]:
    # 每个核的并行任务：(预分词)，encode，然后计数

    with open(input_path, "rb") as f:
        f.seek(start)
        raw_data = f.read(end - start).replace(b"\r\n", b"\n")
        chunk = raw_data.decode("utf-8", errors="ignore")

        # 预分词
        chunk_split = pre_tokenization(chunk, special_tokens)

        # encode每个单词并计数
        word_counts = Counter()
        for word in chunk_split:
            word_b = tuple(word.encode("utf-8", errors="ignore"))
            word_counts[word_b] += 1

        return dict(word_counts)

class BPEtokenizer:
    '''
    输入input_path(包含training data), vocab_size(int), special_tokens(list[str]).

    返回vocab(dict[int, bytes], int为原ID, bytes为token bytes), 

    和merges(list[tuple[bytes, bytes]],记录所有被合并的bytes). 
    '''

    def __init__(self, vocab=None, merges=None, special_tokens=None, pair_counts = Counter()):
        self.vocab = vocab if vocab is not None else {i: bytes([i]) for i in range(256)}
        self.merges = merges if merges is not None else []
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.pair_counts = Counter()
        self.pair_to_words = defaultdict(set)
        self.changed_pairs = set()
        self.candidates = set()
        self.current_max_freq = -1
    
    def counting_init(self, input_path,
         special_tokens, num_chunks=1000
        ) -> dict[tuple[int, ...], int]:
        # 初始化：第一次计数

        # 按边界切割
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f,
                desired_num_chunks = num_chunks,
                split_special_token = special_tokens[0].encode("utf-8")
            )
        