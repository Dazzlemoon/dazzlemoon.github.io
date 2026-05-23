# -*- coding: utf-8 -*-
from pathlib import Path

BLOG = Path(__file__).resolve().parents[1]
src = BLOG.parent / "post-to-wechat" / "sog核介绍" / "dp是否可接入sog核" / "2026-dpa4-sog-longrange-wechat.md"
dst = BLOG / "_posts" / "2026-05-23-DPA4-SOG-长程能否接棒MACE-POLAR-1.md"

raw = src.read_text(encoding="utf-8")
if raw.startswith("---\n"):
    end = raw.find("\n---\n", 4)
    body = raw[end + 5 :].lstrip("\n")
else:
    body = raw

lines = body.splitlines()
i = 0
while i < len(lines) and not lines[i].strip():
    i += 1
if i < len(lines) and lines[i].startswith("# "):
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    body = "\n".join(lines[i:]).lstrip("\n")

replacements = [
    (
        "[`SOG_multipole_kspace_实现与扩展思路.md`](SOG_multipole_kspace_实现与扩展思路.md)",
        "[SOG 多极 k 空间实现说明](https://github.com/Dazzlemoon/sog)",
    ),
    ("## 07. 给公众号读者的一句话总结", "## 07. 一句话总结"),
    ("`dpmd-public-dpa4/doc/model/dpa4.md`", "[DPA4/SeZM 文档](https://github.com/deepmodeling/deepmd-kit)"),
]
for old, new in replacements:
    body = body.replace(old, new)

hero = (
    "![一图总结：DPA4 短程 × SOG 长程]"
    "(/img/posts/dpa4-sog-longrange/dpa4-sog-longrange-onepage.png)\n\n"
)
body = hero + body

extra = """
---

## 08. 延伸阅读（站内）

- [Kim 2026：可极化多极矩长程电静学](/2026/05/16/Kim2026-可极化多极矩长程电静学/)
- [CACE-SOG 阶段进展（water / MAPbI₃）](/2026/05/17/CACE-SOG-阶段进展汇报-water-MAPbI3/)
- [SOG 开源仓库](https://github.com/Dazzlemoon/sog)

"""

if "## 08. 延伸阅读" not in body:
    ref_idx = body.find("\n## 参考")
    if ref_idx != -1:
        body = body[:ref_idx] + extra + body[ref_idx:]

fm = """---
layout:     post
title:      DPA4 短程有多强？SOG 长程能否接棒 MACE-POLAR-1？
subtitle:   DPA4（SeZM）短程强项、Kim 框架下 SOG 相对 LES 的长程优势、与 MACE-POLAR-1 的分场景边界，以及 DPA4+SOG 组合展望
date:       2026-05-23
author:     qqz
header-img: img/post-bg-desk.jpg
catalog:    true
tags:
    - MLIP
    - DPA4
    - SOG
    - 长程电静学
    - 势函数
---

"""

dst.write_text(fm + body + ("\n" if not body.endswith("\n") else ""), encoding="utf-8")
print(f"Wrote {dst}")
