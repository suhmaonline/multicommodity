#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 ModelScope 直接下载 Chronos-T5-small 权重与 T5 tokenizer 到本地目录，
避免 HuggingFace 连接被阻断。模型本身与 amazon/chronos-t5-small 一致。
"""
import os, sys, urllib.request, urllib.parse, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "data", "model_cache", "chronos-t5-small-ms")
os.makedirs(OUT, exist_ok=True)

CHRONOS = "https://www.modelscope.cn/api/v1/models/AI-ModelScope/chronos-t5-small/repo?Revision=master&FilePath="
T5 = "https://www.modelscope.cn/api/v1/models/AI-ModelScope/t5-small/repo?Revision=master&FilePath="

CHRONOS_FILES = [
    "config.json",
    "configuration.json",
    "generation_config.json",
    "model.safetensors",
]
# Chronos 复用 T5 tokenizer；从 t5-small 仓库补齐 tokenizer 文件
T5_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "spiece.model",
]


def download(url, dst, timeout=120):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  [skip] {os.path.basename(dst)} 已存在")
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    with open(dst, "wb") as f:
        f.write(data)
    print(f"  [ok] {os.path.basename(dst)} {len(data)/1e6:.1f} MB")
    return True


def main():
    print(f"[INFO] 输出目录: {OUT}")
    for f in CHRONOS_FILES:
        u = CHRONOS + urllib.parse.quote(f, safe="")
        print(f"[GET] chronos/{f}")
        try:
            download(u, os.path.join(OUT, f))
        except Exception as e:
            print(f"  [ERR] {f}: {type(e).__name__} {e}")
            sys.exit(1)
    for f in T5_FILES:
        u = T5 + urllib.parse.quote(f, safe="")
        print(f"[GET] t5/{f}")
        try:
            download(u, os.path.join(OUT, f))
        except Exception as e:
            print(f"  [ERR] {f}: {type(e).__name__} {e}")
            sys.exit(1)
    print("[DONE] 本地 Chronos 目录准备完成:", OUT)


if __name__ == "__main__":
    main()
