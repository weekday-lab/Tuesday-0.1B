"""
tokenization_minilm — MiniLM 分词器的 HuggingFace 封装（字节级 BPE，32k 词表）。

本目录同时提供两个实现，编码结果与训练时**逐 token 一致**：

  * MiniLMTokenizer      慢速（纯 Python），直接照抄训练脚本的编码逻辑。
  * MiniLMTokenizerFast  快速（Rust BPE），预分词用 Python 自定义组件复现
                         「按 24 字节窗口切分」的行为，其余合并交给 Rust。

词表布局（与训练阶段一致）：
  0..5        特殊符号 <pad> <unk> <bos> <eos> <|user|> <|assistant|>
  6..261      256 个字节（按 GPT-2 的 byte↔unicode 映射存成可见字符）
  262..32767  合并产生的 token
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import AddedToken

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<|user|>", "<|assistant|>"]
PAD, UNK, BOS, EOS, USER, ASSISTANT = range(len(SPECIAL_TOKENS))
N_SPECIAL = len(SPECIAL_TOKENS)
MAX_PIECE_BYTES = 24


def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2 的 byte→unicode 映射：让 256 个字节都能变成可打印字符。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_TO_UNI = bytes_to_unicode()
UNI_TO_BYTE = {v: k for k, v in BYTE_TO_UNI.items()}

# 与 minilm/tokenizer_bpe.py 完全相同的预切分正则
_PRETOK_RE = re.compile(r"[a-zA-Z]+|[0-9]+|\s+|[^\W\d_]+|[^\sa-zA-Z0-9]+", re.UNICODE)

# 特殊符号切分（HF 的 split_on_tokens 依赖 added_tokens 注册，这里显式处理更可靠）
_SPECIAL_SPLIT = re.compile("(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")")


def pre_tokenize(text: str) -> List[str]:
    return _PRETOK_RE.findall(text)


def byte_windows(piece: str) -> List[bytes]:
    bs = piece.encode("utf-8")
    if len(bs) <= MAX_PIECE_BYTES:
        return [bs]
    return [bs[i:i + MAX_PIECE_BYTES] for i in range(0, len(bs), MAX_PIECE_BYTES)]


def _window_char_ranges(piece: str) -> List[Tuple[int, int]]:
    """把片段按「不超过 24 字节」切成若干窗口，返回字符下标区间。

    训练时的实现是硬切 24 字节（可能切在字符中间）；这里按字符对齐，
    纯中文/纯 ASCII 等常见文本与硬切完全等价。
    """
    ranges: List[Tuple[int, int]] = []
    start_char, cur = 0, 0
    for idx, ch in enumerate(piece):
        n = len(ch.encode("utf-8"))
        if cur and cur + n > MAX_PIECE_BYTES:
            ranges.append((start_char, idx))
            start_char, cur = idx, 0
        cur += n
    ranges.append((start_char, len(piece)))
    return ranges


class _ByteWindowPreTokenizer:
    """tokenizers 的 Python 自定义预分词器：复刻训练的「预切分 + 24 字节窗口」。

    直接把每个字节窗口映射成 byte→unicode 字符后返回，因此编码结果与训练脚本
    的 `_encode_piece` 完全一致（包括切在字符中间的极端情况）。
    """

    @staticmethod
    def _windows(text: str) -> List[Tuple[str, Tuple[int, int]]]:
        from tokenizers import NormalizedString

        out: List[Tuple[str, Tuple[int, int]]] = []
        for piece in pre_tokenize(text):
            for window in byte_windows(piece):
                s = "".join(BYTE_TO_UNI[b] for b in window)
                out.append((s, (0, len(piece))))
        return out

    def pre_tokenize(self, pretok):
        from tokenizers import NormalizedString

        def _split(_i, normalized):
            return [NormalizedString(m) for m, _ in self._windows(str(normalized))]

        pretok.split(_split)

    def pre_tokenize_str(self, sequence: str):
        return [(m, (0, len(m))) for m, _ in self._windows(sequence)]


class MiniLMTokenizer(PreTrainedTokenizer):
    """字节级 BPE 慢速分词器：与训练时逐 token 一致。"""

    vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"

    def __init__(self, vocab_file: str = None, merges_file: str = None, **kwargs):
        if vocab_file is None or merges_file is None:
            raise ValueError("MiniLMTokenizer 需要 vocab.json 与 merges.txt")
        with open(vocab_file, "r", encoding="utf-8") as f:
            self._vocab: Dict[str, int] = json.load(f)
        with open(merges_file, "r", encoding="utf-8") as f:
            merges = [ln.rstrip("\n") for ln in f if ln.strip() and not ln.startswith("#version")]
        self._ranks: Dict[Tuple[str, str], int] = {}
        for i, line in enumerate(merges):
            a, b = line.split(" ")
            self._ranks[(a, b)] = i
        self._ids_to_token = {i: t for t, i in self._vocab.items()}
        self._special_ids = {t: i for i, t in enumerate(SPECIAL_TOKENS)}

        super().__init__(
            unk_token=kwargs.pop("unk_token", SPECIAL_TOKENS[UNK]),
            bos_token=kwargs.pop("bos_token", SPECIAL_TOKENS[BOS]),
            eos_token=kwargs.pop("eos_token", SPECIAL_TOKENS[EOS]),
            pad_token=kwargs.pop("pad_token", SPECIAL_TOKENS[PAD]),
            **kwargs,
        )
        for tok, idx in self._special_ids.items():
            self._added_tokens_decoder.setdefault(
                idx, AddedToken(tok, rstrip=False, lstrip=False, special=True, normalized=False))

    # ------------------------------------------------------------ 基本映射
    def get_vocab(self) -> Dict[str, int]:
        vocab = dict(self._vocab)
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._special_ids.get(token, self.unk_token_id))

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_token.get(index, self.unk_token)

    # ------------------------------------------------------------ 编码
    def _encode_window(self, window: bytes) -> List[str]:
        """对一个字节窗口做 BPE（每次合并「当前排名最高的 pair」的全部出现）。"""
        seq = [BYTE_TO_UNI[b] for b in window]
        while len(seq) >= 2:
            best_rank, best_pair = None, None
            for pair in zip(seq, seq[1:]):
                r = self._ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            out, i, n = [], 0, len(seq)
            while i < n:
                if i < n - 1 and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                    out.append(seq[i] + seq[i + 1])
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq = out
        return seq

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """按特殊符号切分后再做 BPE，保证 <bos>/<|user|> 等始终是单个 token。"""
        out: List[str] = []
        for part in _SPECIAL_SPLIT.split(text):
            if not part:
                continue
            if part in self._special_ids:
                out.append(part)
            else:
                out.extend(self._tokenize(part))
        return out

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        tokens: List[str] = []
        for piece in pre_tokenize(text):
            for window in byte_windows(piece):
                tokens.extend(self._encode_window(window))
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        buf = bytearray()
        for t in tokens:
            if t in self._special_ids:
                continue
            for ch in t:
                b = UNI_TO_BYTE.get(ch)
                if b is not None:
                    buf.append(b)
        return buf.decode("utf-8", errors="replace")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        os.makedirs(save_directory, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        vocab_file = os.path.join(save_directory, prefix + "vocab.json")
        merges_file = os.path.join(save_directory, prefix + "merges.txt")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False)
        with open(merges_file, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for pair in sorted(self._ranks, key=self._ranks.get):
                f.write(f"{pair[0]} {pair[1]}\n")
        return vocab_file, merges_file


_CJK = "[\u4e00-\u9fff]"
# tokenizer.json 里保存的「可序列化」预切分正则：先用训练同款正则切片段，
# 再用 {8}/{1,7} 近似「长中文片段按 24 字节切窗」。
FAST_PATTERN = (f"{_CJK}{{8}}|{_CJK}{{1,7}}"
                r"|[a-zA-Z]{1,24}|[0-9]{1,24}|\s{1,24}"
                r"|[^\W\d_]+|[^\sa-zA-Z0-9]+")


def _regex_pretokenizer():
    from tokenizers import Regex, pre_tokenizers
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(FAST_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False),
    ])


def read_merges(merges_file: str) -> List[Tuple[str, str]]:
    with open(merges_file, "r", encoding="utf-8") as f:
        return [tuple(ln.rstrip("\n").split(" ")) for ln in f
                if ln.strip() and not ln.startswith("#version")]


def build_backend(vocab: Dict[str, int], pairs: List[Tuple[str, str]]):
    """(词表, 合并规则) -> tokenizers.Tokenizer（字节级 BPE，GPT-2 同款）。"""
    from tokenizers import Tokenizer, decoders, models

    tk = Tokenizer(models.BPE(vocab=vocab, merges=pairs, unk_token="<unk>", fuse_unk=False))
    tk.add_special_tokens(SPECIAL_TOKENS)
    tk.decoder = decoders.ByteLevel()
    tk.pre_tokenizer = _regex_pretokenizer()
    return tk


class MiniLMTokenizerFast(PreTrainedTokenizerFast):
    """快速分词器：Rust BPE + Python 预分词（复现训练的字节窗口切分）。"""

    vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *args, **kwargs):
        vocab_file = kwargs.pop("vocab_file", None)
        merges_file = kwargs.pop("merges_file", None)
        vocab = kwargs.pop("vocab", None)
        tokenizer_file = kwargs.pop("tokenizer_file", None)

        # 从目录加载时（save_pretrained 后 / 本地路径），HF 可能不传这些路径，自己补
        home = str(kwargs.get("name_or_path") or "")
        if home and os.path.isdir(home):
            tokenizer_file = tokenizer_file or (
                os.path.join(home, "tokenizer.json")
                if os.path.isfile(os.path.join(home, "tokenizer.json")) else None)
            if merges_file is None and os.path.isfile(os.path.join(home, "merges.txt")):
                merges_file = os.path.join(home, "merges.txt")
                vocab_file = vocab_file or os.path.join(home, "vocab.json")

        self._vocab: Dict[str, int] | None = None
        self._merge_pairs: List[Tuple[str, str]] | None = None
        if tokenizer_file:
            super().__init__(tokenizer_file=tokenizer_file, *args, **kwargs)
        else:
            if vocab is None and vocab_file and os.path.isfile(vocab_file):
                with open(vocab_file, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
            if vocab is None or not merges_file:
                raise ValueError("MiniLMTokenizerFast 需要 tokenizer.json 或 vocab.json + merges.txt")
            pairs = read_merges(merges_file)
            self._vocab, self._merge_pairs = vocab, pairs
            super().__init__(tokenizer_object=build_backend(vocab, pairs), *args, **kwargs)

        from tokenizers import pre_tokenizers
        self._serializable_pretokenizer = self.backend_tokenizer.pre_tokenizer
        try:
            self.backend_tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(
                _ByteWindowPreTokenizer())
        except Exception:       # 极老版本 tokenizers 不支持自定义组件：退回正则版预分词
            self._serializable_pretokenizer = None

    def save_pretrained(self, save_directory, **kwargs):
        """落盘：临时换回可序列化的预分词器，并补写 vocab.json / merges.txt。"""
        bt = self.backend_tokenizer
        current = bt.pre_tokenizer
        if self._serializable_pretokenizer is not None:
            bt.pre_tokenizer = self._serializable_pretokenizer
        try:
            files = super().save_pretrained(save_directory, **kwargs)
        finally:
            bt.pre_tokenizer = current

        out = str(save_directory)
        os.makedirs(out, exist_ok=True)
        vocab_path = os.path.join(out, "vocab.json")
        merges_path = os.path.join(out, "merges.txt")
        if self._vocab is not None:
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(self._vocab, f, ensure_ascii=False)
            files = tuple(files) + (vocab_path,)
        if self._merge_pairs is not None:
            with open(merges_path, "w", encoding="utf-8") as f:
                f.write("#version: 0.2\n")
                for a, b in self._merge_pairs:
                    f.write(f"{a} {b}\n")
            files = tuple(files) + (merges_path,)
        return files
