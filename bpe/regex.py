import regex as re
from .base import Tokenizer, get_stats, merge


#不同的字符切分正则
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 这里会把文本做切分然后再训练，例如把英文、数字、表情进行预切分，这样可以避免影响bpe的训练
        text_chunks = re.findall(self.compiled_pattern, text)


        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # 这里的逻辑和basic.py基本相同
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def register_special_tokens(self, special_tokens):
        # 注册特殊字符，可以看看FIM技术
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        可以处理特殊字符，例如# 场景：用户输入
        user_input = "Hello<|endoftext|>malicious code"，如果没有处理特殊字符，就会把<|endoftext|>当作普通字符看待，这就可能
        这里会出现安全问题，攻击者可以在输入中注入特殊 token，破坏模型的行为（例如提前结束生成）。
        "all"允许所有特殊 token
        "none"不允许特殊 token
        "none_raise"不允许，且检查
        set(...)只允许指定的
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text)
        #把特殊字符构建为正则表达式 ["<|endoftext|>", "<|start|>"]  变为(<\|endoftext\|>|<\|start\|>)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        #根据特殊字符进行切分text = "Hello<|endoftext|>World<|start|>Foo" 切分为
        """
        [
            'Hello',  # 第 1 部分：普通文本
            '<|endoftext|>',  # 第 2 部分：特殊 token（被捕获）
            'World',  # 第 3 部分：普通文本
            '<|start|>',  # 第 4 部分：特殊 token（被捕获）
            'Foo'  # 第 5 部分：普通文本
        ]
        """
        special_chunks = re.split(special_pattern, text)

        ids = []
        #对正常字符和特殊字符分别进行编码
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
