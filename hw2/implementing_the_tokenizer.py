import numpy as np
from typing import Dict, List, Set, Tuple, Iterable, Iterator
import regex
import json
import tiktoken

def gpt2_bytes_to_unicode_local(): # 
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
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class implement_bpe_tokenizer:
    # 将字节转换为Unicode字符,调用函数直接就返回字典{数字:unicode字符}结果
    _BYTES_TO_UNICODE_MAP = gpt2_bytes_to_unicode_local()
    # 将Unicode字符转换为字节,调用函数直接就返回字典{unicode字符:字节}结果
    _UNICODE_TO_BYTES_MAP = {v: bytes([k]) for k, v in _BYTES_TO_UNICODE_MAP.items()}
    
    def __init__(self,vocab,merges,special_token=None):
        """
        构造函数：从给定词汇表、合并规则和特殊符号创建分词器。

        :param vocab: 词汇表，键为整数ID，值为对应的字节串。
        :param merges: 合并规则列表，每个元素是一个元组(bytes_token1, bytes_token2)。
                       这些规则应该按训练时的优先级排序。
        :param special_tokens: 可选的特殊符号列表（字符串形式）。
        """
        self.vocab: Dict[int, bytes] = vocab
        self.merges:List[Tuple[bytes, bytes]] = merges
        self.special_tokens: List[bytes] = [] # 存储特殊符号的字节表示
        # 用于高效查找的逆词汇表
        self._bytes_to_id: Dict[bytes, int] = {token: id for id, token in vocab.items()}

        # 找到当前词汇表中最大的ID，用于为新添加的特殊符号分配ID
        self._next_id = max(vocab.keys()) + 1 if vocab else 0
