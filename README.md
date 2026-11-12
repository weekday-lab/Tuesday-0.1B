---
license: other
language:
  - zh
library_name: transformers
pipeline_tag: text-generation
tags:
  - minilm
  - causal-lm
  - chinese
  - from-scratch
  - text-generation
  - dpo
---

# Tuesday-0.1b（V6）

Tuesday-0.1b 是一个 **1.1 亿参数的中文因果语言模型**，从零自研、不依赖任何开源预训练权重，
结构为 12 层 Pre-LayerNorm Transformer 解码器。本仓库为 **V6 系列**，模型目录本身就是标准
HuggingFace 仓库：`config.json`、`model.safetensors`、分词器全套文件、`chat_template.jinja`
与动态模型代码**一次性齐备**，`trust_remote_code=True` 直接加载，无需任何转换脚本。

> **本仓库版本：`sft_hf_v6_dpo`（偏好对齐版，推荐日常对话使用）**

## 版本对照

同一套底座派生三个版本，选一个下载即可：

| 版本 | 目录名 | 说明 | 适用场景 |
|---|---|---|---|
| 基座 | `pretrain_hf_v6` | 纯语言建模基座，未做指令微调 | 继续预训练 / 微调的起点 |
| 指令版 | `sft_hf_v6` | 指令微调后，能遵循对话格式 | 需要原始 SFT 行为的场景 |
| **对齐版** | **`sft_hf_v6_dpo`** | 在指令版上做偏好对齐，回答更简洁、复读更少 | **直接对话（推荐）** |

三者架构与词表完全一致，可互换使用；权重各约 423MB（F32）。

## 模型结构

| 项目 | 值 |
|---|---|
| 参数量 | 110.9M（tied embedding 只算一次） |
| 层数 / 隐藏维度 / 头数 | 12 / 768 / 12（head_dim 64） |
| FFN 维度 | 3072，GELU |
| 位置编码 | 可学习绝对位置编码，最大长度 1024 |
| 归一化 | Pre-LayerNorm（eps = 1e-5） |
| 词表 | 32,768，字节级 BPE，中文约 1.44 字/token |
| 权重精度 | float32（`model.safetensors`，443,714,848 字节） |
| 特殊 token | `<pad>=0`、`<unk>=1`、`<bos>=2`、`<eos>=3` |

## 快速开始

```bash
pip install "transformers>=5.0.0" torch safetensors tokenizers
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "Tuesday-0.1b"                      # 本仓库目录 / HF repo id
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, dtype=torch.float32).eval()

messages = [{"role": "user", "content": "用一句话介绍你自己。"}]
enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                              return_tensors="pt")
# transformers 5.x 返回 BatchEncoding，需取 ["input_ids"]；4.x 直接是 Tensor
input_ids = enc["input_ids"] if hasattr(enc, "keys") else enc

with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=128, do_sample=True,
                         temperature=0.7, top_p=0.95, top_k=50,
                         repetition_penalty=1.05,
                         pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id)

print(tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True))
```

多轮对话直接把历史消息按顺序追加到 `messages` 即可（`system` 角色会被拼接到第一条 user 前）。

## 对话模板

`chat_template.jinja` 已随仓库提供，`apply_chat_template` 会自动使用。手工拼接等价形式：

```text
<bos><|user|>{用户输入}<|assistant|>{模型回答}<eos><|user|>{下一轮输入}<|assistant|>
```

要点：

- 序列以 `<bos>` 开头，模型回答以 `<eos>` 结尾（推理时遇到 `<eos>` 即停）。
- 生成时提示词末尾必须带 `<|assistant|>`（`add_generation_prompt=True` 会自动加）。
- 上下文上限 **1024 token**，超出请截断历史，否则位置编码越界。

## 生成参数

仓库内 `generation_config.json` 已给出推荐默认值，直接 `generate()` 即可：

| 参数 | 默认值 |
|---|---|
| `do_sample` / `temperature` | `true` / `0.7` |
| `top_p` / `top_k` | `0.95` / `50` |
| `repetition_penalty` | `1.05` |
| `max_new_tokens` | `256` |

小模型对解码策略较敏感：**贪心解码（`do_sample=False`）在个别问题上容易陷入重复**，
日常使用建议保持采样 + `repetition_penalty ≥ 1.05`。需要确定性输出时，请配合
`no_repeat_ngram_size=3` 使用。

## 文件清单

| 文件 | 说明 |
|---|---|
| `model.safetensors` | 模型权重（F32，423MB） |
| `config.json` | 结构配置，含 `auto_map`（指向动态代码） |
| `generation_config.json` | 推荐生成参数 |
| `tokenizer.json` | 快速分词器（Rust 格式，2.7MB） |
| `vocab.json` / `merges.txt` | BPE 词表与合并规则 |
| `tokenizer_config.json` / `special_tokens_map.json` | 分词器配置与特殊 token |
| `chat_template.jinja` | 对话模板 |
| `configuration_minilm.py` / `modeling_minilm.py` / `tokenization_minilm.py` | 动态加载所需的自定义代码 |

## 自检：一条命令验证模型可用

仓库附带 `test_model.py`，对下载后的模型做四层检查（文件完整性 → 分词器与模板 →
前向 logits 健康 → 生成非空 / 非复读）：

```bash
python test_model.py --model ./                # 交互式查看
python test_model.py --model ./ --strict       # 发布前 / CI 闸门，失败返回非 0
python test_model.py --model ./ --prompt "写一句诗"
```

实测输出：

```text
[1/4] 文件完整性
  [ OK ] model.safetensors 423MB
  [ OK ] 10 个配置文件齐全
[2/4] 分词器 / 对话模板
  [ OK ] 分词器加载成功 vocab=32768
  [ OK ] 对话模板正常：'<bos><|user|>你好<|assistant|>'
[3/4] 前向计算
  [ OK ] 模型加载成功 params=110.9M dtype=torch.float32
  [ OK ] 前向正常 logits=(1, 4, 32768) last-token max=15.78
[4/4] 生成质量冒烟
  [ OK ] 用一句话介绍你自己。 -> 我是一个人工智能语言模型，没有情感和偏好……
  [ OK ] 帮我写一句鼓励学习的话。 -> 学习是人生中最大的财富。
========================================================
结论：全部通过，模型可正常加载与推理
```

完整源码：

<details>
<summary>test_model.py（点击展开）</summary>

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tuesday-0.1b 模型冒烟测试（发布自检）

用法：
    python test_model.py --model ./              # 交互式查看，输出示例回答
    python test_model.py --model ./ --strict      # CI/发布前闸门，任一检查失败退出码 1
    python test_model.py --model ./ --prompt "写一句诗"

依赖：transformers>=5.0.0, torch, safetensors, tokenizers
"""
import argparse
import sys
from pathlib import Path

REQUIRED_FILES = [
    "config.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "tokenizer.json", "chat_template.jinja",
    "configuration_minilm.py", "modeling_minilm.py", "tokenization_minilm.py",
]
DEFAULT_PROMPTS = [
    "用一句话介绍你自己。",
    "水的沸点是多少度？",
    "帮我写一句鼓励学习的话。",
]


def fail(msg):
    print(f"  [FAIL] {msg}")
    return False


def ok(msg):
    print(f"  [ OK ] {msg}")
    return True


def check_files(root: Path):
    """1) 文件完整性：权重、分词器、自定义代码是否齐全（排除半截下载）"""
    print("[1/4] 文件完整性")
    good = True
    for name in REQUIRED_FILES:
        p = root / name
        if not p.is_file():
            good = fail(f"缺少 {name}") and good
    weights = list(root.glob("*.safetensors"))
    if not weights and not (root / "pytorch_model.bin").is_file():
        good = fail("缺少模型权重（*.safetensors）") and good
    for w in weights:
        size = w.stat().st_size
        if size < 100 * 1024 * 1024:
            good = fail(f"{w.name} 仅 {size/1e6:.1f}MB，疑似下载不完整") and good
        else:
            ok(f"{w.name} {size/1024/1024:.0f}MB")
    if good:
        ok(f"{len(REQUIRED_FILES)} 个配置文件齐全")
    return good


def check_tokenizer(root: Path):
    """2) 分词器与对话模板可用"""
    print("[2/4] 分词器 / 对话模板")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        return fail(f"分词器加载失败: {e}"), None
    ok(f"分词器加载成功 vocab={len(tok)}")

    msgs = [{"role": "user", "content": "你好"}]
    try:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt")
    except Exception as e:  # noqa: BLE001
        return fail(f"chat_template 渲染失败: {e}"), tok
    # transformers 5 返回 BatchEncoding，4.x 返回 Tensor
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    text = tok.decode(ids[0] if hasattr(ids, "shape") and ids.dim() > 1 else ids,
                      skip_special_tokens=False)
    expect = ["<bos>", "<|user|>", "你好", "<|assistant|>"]
    missing = [t for t in expect if t not in text]
    if missing:
        return fail(f"模板渲染结果缺少标记 {missing}: {text!r}"), tok
    ok(f"对话模板正常：{text!r}")
    return True, tok


def check_forward(root: Path, tok):
    """3) 前向计算：logits 数值健康"""
    print("[3/4] 前向计算")
    import torch
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(str(root), trust_remote_code=True)
        model.eval()
    except Exception as e:  # noqa: BLE001
        return fail(f"模型加载失败: {e}"), None
    n_param = sum(p.numel() for p in model.parameters())
    ok(f"模型加载成功 params={n_param/1e6:.1f}M dtype={next(model.parameters()).dtype}")

    enc = tok.apply_chat_template([{"role": "user", "content": "你好"}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    with torch.no_grad():
        logits = model(ids).logits
    if not torch.isfinite(logits).all():
        return fail("logits 含 NaN/Inf"), None
    ok(f"前向正常 logits={tuple(logits.shape)} "
       f"last-token max={logits[0, -1].max().item():.2f}")
    return True, model


def check_generate(model, tok, prompts, max_new_tokens):
    """4) 生成：非空、非复读"""
    print("[4/4] 生成质量冒烟")
    import torch
    good = True
    for q in prompts:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        toks = ans.split()
        degenerate = len(toks) >= 6 and len(set(toks)) <= max(2, len(toks) // 6)
        if not ans:
            good = fail(f"空回答：{q}") and good
        elif degenerate:
            good = fail(f"疑似复读：{q} -> {ans[:60]!r}") and good
        else:
            ok(f"{q} -> {ans[:60]}{'…' if len(ans) > 60 else ''}")
    return good


def main():
    ap = argparse.ArgumentParser(description="Tuesday-0.1b 冒烟测试")
    ap.add_argument("--model", default="./", help="模型目录（含 config.json 的目录）")
    ap.add_argument("--prompt", action="append", default=None,
                    help="自定义测试问题，可重复；不传则用内置三问")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--strict", action="store_true",
                    help="严格模式：任何检查失败返回非 0（发布前/CI 用）")
    args = ap.parse_args()

    root = Path(args.model).expanduser().resolve()
    print(f"模型目录：{root}")
    if not root.is_dir():
        sys.exit(f"目录不存在：{root}")

    results = [check_files(root)]
    r_tok, tok = check_tokenizer(root)
    results.append(r_tok)
    if tok is not None:
        r_fwd, model = check_forward(root, tok)
        results.append(r_fwd)
        if model is not None:
            results.append(check_generate(model, tok,
                                          args.prompt or DEFAULT_PROMPTS,
                                          args.max_new_tokens))
    passed = all(results)
    print("=" * 56)
    print("结论：全部通过，模型可正常加载与推理" if passed
          else "结论：存在失败项，请检查上方 [FAIL]")
    print("=" * 56)
    sys.exit(0 if passed or not args.strict else 1)


if __name__ == "__main__":
    main()
```

</details>

## 硬件与环境

- 权重 F32 共 423MB，**CPU 即可运行**（单条 128 token 回答通常为秒级）；GPU 4GB 以上更流畅。
- 依赖：`transformers>=5.0.0`（验证版本 5.22.1）、`torch>=2.0`、`safetensors`、`tokenizers>=0.23`。
- **必须** `trust_remote_code=True`：模型使用自定义代码（`minilm` 结构），不是 transformers 内置架构。
- 半精度推理：把 `dtype=torch.float32` 换成 `torch.bfloat16` 并配 GPU 即可，权重会自动转换。

## 已知局限

- **参数规模仅 0.11B**：事实性知识薄弱，可能给出错误事实（例如把水的沸点说成 4.8 度）。不要把本模型用作事实查询或专业知识问答的可信来源。
- **上下文 1024 token**：长文摘要、多轮长对话会被截断。
- **中文为主**：语料以中文为核心，英文能力有限。
- **可能复读或自我重复**：小模型的典型现象，采样 + `repetition_penalty` 可显著缓解；贪心解码时更明显。
- **未做 RLHF 级安全对齐**：生产环境请自行加内容过滤与业务校验。

## 数据与许可

- 模型权重与推理代码：**other**。
- 指令微调数据基于公开中文指令数据集（BELLE 系列）与公开中文语料构建；使用与再分发请同时遵守上游数据集许可。
- 本仓库仅包含推理所需文件，不含训练数据。

## 引用

```bibtex
@misc{tuesday_0_1b_v6,
  title  = {Tuesday-0.1b: a 110M-parameter Chinese causal language model},
  author = {Tuesday-0.1b contributors},
  year   = {2026},
  note   = {Model version V6}
}
```
