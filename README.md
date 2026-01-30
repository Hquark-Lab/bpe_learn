# 1.项目的基本结构
 base.ipynb → bpe的接口设计
 basic.ipynb → bpe算法的核心实现
 regex.ipynb → 带预处理的核心实现
 gpt4.ipynb → 借助tiktoken实现的一个生产级tokenizer

# 2.编码的理解
ch = '🌞'  这个字符的 Unicode 码点是 U+1F31E，编码为 UTF-8 需要 4 个字节
print(ch.encode("utf-8"))
打印结果：b'\xf0\x9f\x8c\x9e'

ch = '🌞'
print(ord(ch))        # 127774（十进制）
print(hex(ord(ch)))   # 0x1f31e（十六进制）

b''：表示这是一个 bytes 对象（字节串）
\xf0：十六进制 F0，十进制 240
\x9f：十六进制 9F，十进制 159
\x8c：十六进制 8C，十进制 140
\x9e：十六进制 9E，十进制 158

所谓的编码就是针对Unicode的码点进行编码。

## 2.1如何从`b'\xf0\x9f\x8c\x9e'` 换算为 127774 

### 已知条件：
UTF-8 字节序列：`b'\xf0\x9f\x8c\x9e'`  
十六进制：`F0 9F 8C 9E`  
二进制：`11110000 10011111 10001100 10011110`

### 步骤1：识别 UTF-8 编码模式

查看第一个字节 `F0` (二进制 `11110000`)：
- 以 `11110` 开头 → 这是 **4 字节 UTF-8 编码模式**
- UTF-8 四字节模式模板：`11110xxx 10xxxxxx 10xxxxxx 10xxxxxx`

### 步骤2：提取有效数据位
```
字节1: 11110000 (F0)   → 模板: 11110xxx → 有效位: xxx = 000
字节2: 10011111 (9F)   → 模板: 10xxxxxx → 有效位: xxxxxx = 011111
字节3: 10001100 (8C)   → 模板: 10xxxxxx → 有效位: xxxxxx = 001100  
字节4: 10011110 (9E)   → 模板: 10xxxxxx → 有效位: xxxxxx = 011110
```
提取的有效位：
```
字节1: 000
字节2: 011111
字节3: 001100
字节4: 011110
```

### 步骤3：组合有效位形成 Unicode 码点

四字节 UTF-8 编码格式：
```
11110aaa 10bbbbbb 10cccccc 10dddddd
Unicode 码点: 000aaabbbbbbccccccdddddd
```

将提取的位按格式组合：
```
a = 000    (3 bits)
b = 011111 (6 bits)  
c = 001100 (6 bits)
d = 011110 (6 bits)

组合: 000 + 011111 + 001100 + 011110
二进制: 000011111001100011110
十六进制: 0001 1111 0011 0001 1110
分组: 0001 1111 0011 0001 1110 → 0x1F31E
```

### 步骤4：计算十进制值

```
十六进制: 1F31E
十进制计算:
1 × 16⁴ = 1 × 65536 = 65536
F × 16³ = 15 × 4096 = 61440
3 × 16² = 3 × 256   = 768
1 × 16¹ = 1 × 16    = 16
E × 16⁰ = 14 × 1    = 14
总和: 65536 + 61440 + 768 + 16 + 14 = 127774
```

## Python 代码验证

```python
# 方法1：直接解码
utf8_bytes = b'\xf0\x9f\x8c\x9e'
decoded_char = utf8_bytes.decode('utf-8')
print(f"解码后的字符: {decoded_char}")                # 🌞
print(f"Unicode 码点: {ord(decoded_char)}")          # 127774
print(f"Unicode 十六进制: U+{ord(decoded_char):04X}") # U+1F31E

# 方法2：手动计算
def utf8_to_codepoint(byte_data):
    """手动实现 UTF-8 解码"""
    if not byte_data:
        return None
    
    b1 = byte_data[0]
    
    if b1 < 0x80:  # 1字节
        return b1
    elif b1 < 0xE0:  # 2字节
        b2 = byte_data[1]
        return ((b1 & 0x1F) << 6) | (b2 & 0x3F)
    elif b1 < 0xF0:  # 3字节
        b2, b3 = byte_data[1], byte_data[2]
        return ((b1 & 0x0F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)
    else:  # 4字节
        b2, b3, b4 = byte_data[1], byte_data[2], byte_data[3]
        return ((b1 & 0x07) << 18) | ((b2 & 0x3F) << 12) | ((b3 & 0x3F) << 6) | (b4 & 0x3F)

# 计算
utf8_bytes = b'\xf0\x9f\x8c\x9e'
codepoint = utf8_to_codepoint(utf8_bytes)
print(f"\n手动计算结果:")
print(f"十六进制: 0x{codepoint:X}")  # 0x1F31E
print(f"十进制: {codepoint}")       # 127774

# 方法3：使用 int.from_bytes
codepoint = int.from_bytes(utf8_bytes, 'big')
print(f"\n使用 int.from_bytes: {codepoint}")  # 4036003102（注意：这是4字节整数值，不是Unicode码点）
print(f"需要正确解码才能得到 Unicode 码点")
```

# 3.为什么tokennizer要处理一些自定义的字符
在regex.ipynb当中有一个存储special_tokens的集合，在gpt4.ipynb当中定义一部分
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
} 注意gpt4的 Embedding 层有 100277 个 token

**这些有什么用处？** 
例如<|fim_prefix|>  <|fim_middle|>  <|fim_suffix|> 三者组成的Fill-in-Middle (FIM) 技术
我们在cursor中写代码
def calculate_sum(numbers):
    # 光标在这里，你按了 Tab 要求补全
    return total
**传统 GPT 的问题**：
- 它只看到前文：`"def calculate_sum(numbers):\n    "`
- 它不知道你后面有 `return total`
- 可能生成不相关的代码

**FIM 的优势**：
- 它看到前文：`"def calculate_sum(numbers):\n    "`
- **也看到后文**：`"\n    return total"`
- 知道需要计算 `total` 变量
- 生成：`"total = sum(numbers)"`

实际上，就是把那些代码换了中方式拼接起来喂给大模型

把文本重新排列：**前缀 + 后缀 + 中间**

prefix = """def hello():
    """

middle = """name = 'world'
    print(f'Hello, {name}!')
    """

suffix = """return 'done'
"""

fim_text = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"

**大模型是怎么识别这种模式**