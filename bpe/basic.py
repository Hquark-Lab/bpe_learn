from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes) #编码后创建list
        merges = {}  # 创建合并规则
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            # 找到pair出现次数最多的
            pair = max(stats, key=stats.get)
            idx = 256 + i#这里为最新的合并创建一个idx，例如a,a出现次数最多，现在就给aa设置一个编码257
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)#返回合并后的结果
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

            # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


    """
    把文本编码成 token IDs
    def encode(self, text):
    """
    def encode(self,text):
        text_bytes = text.encode("utf-8")  #
        ids = list(text_bytes)
        #每一次循环都要生成最新的stats，每次都是合并之后的
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            """
            # 假设训练时的 merges
            self.merges = {
                (108, 108): 256,  # 第 1 次合并：'ll'
                (104, 101): 257,  # 第 2 次合并：'he'
                (101, 108): 258,  # 第 3 次合并：'el'
                # (108, 111) 没有训练过
            }
            
            # 当前的 stats
            stats = {
                (104, 101): 1,  # 'he'
                (101, 108): 1,  # 'el'
                (108, 108): 1,  # 'll'
                (108, 111): 1   # 'lo' ← 没在 merges 中
            }
            
            # 计算每个 pair 的优先级
            (104, 101) → self.merges.get((104, 101), inf) = 257
            (101, 108) → self.merges.get((101, 108), inf) = 258
            (108, 108) → self.merges.get((108, 108), inf) = 256  ← 最小！
            (108, 111) → self.merges.get((108, 111), inf) = inf
            
            # min() 选择优先级数字最小的
            pair = (108, 108)  # ID 256，优先级最高
            
            这里就是为了在当前的stats中找到在merges当中合并优先级最高的pair
             """
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break  # nothing else can be merged anymore

            idx = self.merges[pair]
            #找到有限级最高的merge，然后进行合并
            ids = merge(ids, pair, idx)
        return ids