#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
 是导入 Python 标准库中的 Unicode 数据库模块
 # 判断字符类型
unicodedata.category('A')      # 'Lu'（大写字母）
unicodedata.category('a')      # 'Ll'（小写字母）
unicodedata.category('1')      # 'Nd'（十进制数字）
"""
import unicodedata


# In[2]:


def get_stats(ids, counts=None):
    """
    [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # 巧妙的方式获得相邻字节
        counts[pair] = counts.get(pair, 0) + 1 #counts.get(pair, 0) pair不在counts中返回0
    return counts


# In[4]:


def merge(ids, pair, idx):
    """
    通过get_stats返回相邻pair，找到出现频率最高的相邻pair，把这种相邻pair合并，
    合并后就属于一项merge规则
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):

        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


# In[5]:


"""
# Token 可能包含各种字节
token = b'\n'        # 换行符
token = b'\x00'      # NULL 字符
token = b'\t'        # Tab
token = b'\x1b[31m'  # ANSI 转义码

# 如果直接打印
print(token.decode('utf-8'))
# 输出：会真的换行！会破坏终端显示！
例如 "Hello\nWorld"  经过处理得到  Hello\u000aWorld  ← \n 被转义成 \u000a
"""

def replace_control_characters(s: str) -> str:
    # 我们不想打印控制字符
    # 因为它们会扭曲输出（比如 \n 或者更糟的）
    
    chars = []
    for ch in s:
        # 检查字符的 Unicode 类别
        if unicodedata.category(ch)[0] != "C":
            # 类别不是 "C" (Control) → 正常字符
            chars.append(ch)
        else:
            # 类别是 "C" → 控制字符，转义显示
            chars.append(f"\\u{ord(ch):04x}")
    
    return "".join(chars)


# In[6]:


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s


# In[7]:


class Tokenizer:

    def __init__(self):

        self.merges = {} # (int, int) -> int 合并规则
        self.pattern = "" # 正则，用于切分字符
        self.special_tokens = {} # str -> int, 用于特殊token切分 {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # 接口
        raise NotImplementedError

    def encode(self, text):
      
        raise NotImplementedError

    def decode(self, ids):
      
        raise NotImplementedError

    def _build_vocab(self):
        # vocab 是字节和码点的映射
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # 遍历merge规则，把需要合并的组合合并起来存储vocab当中
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # 对于特殊token也要存储vocab当中
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        形成两个文件: file_prefix.vocab and file_prefix.model
        1. my_tokenizer.model  → 给机器用（程序加载）
        2. my_tokenizer.vocab  → 给人类用（查看、调试）

        my_tokenizer.model 例子
        bpe v1
        '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
        1
        <|endoftext|> 261
        108 108
        101 108
        111 32
        119 111
        32 119

        my_tokenizer.vocab例子
        [\u0000] 0
        [\u0001] 1
        ...
        [e] 101
        ...
        [l] 108
        ...
        [o] 111
        ...
        [w] 119
        ...
        [\u0020] 32
        ...
        [\u00ff] 255
        [l][l] -> [ll] 256
        [e][l] -> [el] 257
        [o][\u0020] -> [o ] 258
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # 写入版本号
            f.write("bpe v1\n")
            f.write(f"{self.pattern}\n")#这是bpe的核心，切分正则表达式
            # 特殊token
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # 把merge规则也写入
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """按照.model文件的写入的格式，把相关信息加载出来"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "bpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()


# In[ ]:




