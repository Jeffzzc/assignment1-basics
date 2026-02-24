import os
import collections
from typing import List, Tuple, Dict, Set
import re
import json

def gpt2_bytes_to_unicode_local():
    """
    将字节转换为Unicode字符,调用函数直接就返回字典{数字:unicode字符}结果
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    ch = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_stats(token_sequences: List[List[str]]) -> collections.Counter:
    """
    统计token序列中所有相邻unicode字符对的频率
    """
    pair_counts = collections.Counter()
    for sequence in token_sequences:
        for i in range(len(token_sequences) - 1):
            pair = (sequence[i], sequence[i+1])
            pair_counts[pair] += 1
    return pair_counts

def merge_pair_in_sequences(
    token_sequences: List[List[str]],
    pair_to_merge: Tuple[str, str],
    new_token_representation: str
) -> List[List[str]]:
    """
    在token序列中用新的unicode字符表示替换指定的字节对
    用 new_token_representation 替换所有出现的 pair_to_merge。
    假设：
    token_sequences = [['h', 'e', 'l', 'l', 'o']]
    pair_to_merge = ('l', 'l')
    new_token_representation = 'll'
    处理过程：
    遍历 ['h', 'e', 'l', 'l', 'o']
    前两个字节不是 ('l', 'l')，跳过
    到了下标2和3，发现是 ('l', 'l')，合并成 'll'
    结果变成 ['h', 'e', 'll', 'o']
    """