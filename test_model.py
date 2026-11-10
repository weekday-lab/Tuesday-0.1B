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
