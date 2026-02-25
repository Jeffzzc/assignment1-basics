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
    
    def __init__(self,vocab,merges,special_tokens=None):
        """
        构造函数：从给定词汇表、合并规则和特殊符号创建分词器。

        :param vocab: 词汇表，键为整数ID，值为对应的字节串。
        :param merges: 合并规则列表，每个元素是一个元组(bytes_token1, bytes_token2)。
                       这些规则应该按训练时的优先级排序。
        :param special_tokens: 可选的特殊符号列表（字符串形式）。
        """
        self.vocab: Dict[int, bytes] = vocab
        self.merges: List[Tuple[bytes, bytes]] = merges
        self._special_tokens_bytes: List[bytes] = [] # 存储特殊符号的字节表示
        # 用于高效查找的逆词汇表
        self._bytes_to_id: Dict[bytes, int] = {token: id for id, token in vocab.items()}

        # 找到当前词汇表中最大的ID，用于为新添加的特殊符号分配ID
        self._next_id = max(vocab.keys()) + 1 if vocab else 0

        # 处理特殊符号
        if special_tokens:
            for st_str in special_tokens:
                st_bytes = st_str.encode('utf-8')  # 将特殊符号转换为字节串
                self._special_tokens_bytes.append(st_bytes)
                if st_bytes not in self._bytes_to_id:
                    # 如果特殊符号不在词汇表中，则添加它
                    self.vocab[self._next_id] = st_bytes
                    self._bytes_to_id[st_bytes] = self._next_id
                    self._next_id += 1
        
        # 为了在BPE编码时高效查找合并规则，创建一个索引到原始merges列表的字典
        # 键是 (bytes1, bytes2)，值是该合并规则在 _merges 列表中的索引
        self._merges_priority_map: Dict[Tuple[bytes, bytes], int] = {
            merge_pair: i for i, merge_pair in enumerate(self.merges)
        }

        # 构建用于特殊符号切分的正则表达式（字符串形式，为了兼容re模块的字符串输入）
        # 需要确保特殊符号从长到短排序，以避免短符号先匹配长符号的情况
        if self._special_tokens_bytes:
            # 需要将 bytes 类型的特殊符号解码为 str 才能被 regex.escape 处理
            # 并按长度降序排序，以确保长特殊符号优先匹配
            sorted_special_token_strings = sorted(
                [s.decode('utf-8', errors="ignore") for s in self._special_tokens_bytes],
                key=len,
                reverse=True
            )
            # 使用 regex.escape 来正确处理特殊符号中的正则表达式特殊字符
            self._special_tokens_pattern = regex.compile(
                '(' + '|'.join(regex.escape(s) for s in sorted_special_token_strings) + ')'
            )
        else:
            self._special_tokens_pattern = None
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        类方法：从序列化的词汇表和合并列表文件构造并返回一个分词器实例。

        :param vocab_filepath: 词汇表文件的路径。格式通常是 'id token_string'，其中token_string是UTF-8字符串。
        :param merges_filepath: 合并规则文件的路径。格式通常是 'token1 token2'。
        :param special_tokens: 可选的特殊符号列表（字符串形式）。
        :return: 一个 Tokenizer 实例。
        """
        vocab: Dict[int, bytes] = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
            for token_str_repr, token_id in vocab_json.items():
                