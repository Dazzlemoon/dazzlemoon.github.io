# -*- coding: utf-8 -*-
import re
import shutil
from pathlib import Path

BLOG = Path(__file__).resolve().parents[1]
POSTS = BLOG / "_posts"
IMG = BLOG / "img" / "posts"
WECHAT = BLOG.parent / "post-to-wechat"

POSTS_CONFIG = [
    {
        "src": WECHAT / "电池领域思路" / "Kim界面文章介绍" / "公众号稿-Kim2024-LLZO-LCO界面MLIP.md",
        "slug": "kim2024-llzo",
        "filename": "2026-05-20-Kim2024-LLZO-LCO界面MLIP降解机制.md",
        "date": "2026-05-20",
        "title": "全固态电池界面会怎么坏？Kim 2024 用机器学习势「看见」LLZO|LCO 降解全过程",
        "subtitle": "美国 Livermore 实验室用专用 MLIP 做万原子、纳秒级界面模拟，讲清 Li 贫/富如何左右互混、Co 如何钻进石榴石并在晶界成团",
        "tags": ["固态电池", "MLIP", "LLZO", "界面"],
        "image_dirs": [
            (WECHAT / "电池领域思路" / "Kim界面文章介绍" / "images", None),
        ],
        "replacements": [(r"images/", "/img/posts/kim2024-llzo/")],
    },
    {
        "src": WECHAT / "电池领域思路" / "谱学知识" / "公众号稿-红外拉曼声子谱-原理数学与电池应用.md",
        "slug": "ir-raman-phonon",
        "filename": "2026-05-18-红外拉曼声子谱-原理数学与电池应用.md",
        "date": "2026-05-18",
        "title": "红外光谱、拉曼光谱与声子谱：原理、数学与电池研究中的应用",
        "subtitle": "一文讲清 IR、Raman 与声子谱的关系：从选择定则与动力学矩阵出发，落到电池研究中的实际问题与工作流",
        "tags": ["红外光谱", "拉曼光谱", "声子谱", "电池", "谱学"],
        "image_dirs": [],
        "replacements": [],
    },
    {
        "src": WECHAT
        / "电池领域思路"
        / "Zhou2025-非环境热力学与LPSC_LLZO谱学路线"
        / "公众号稿-Zhou2025-非环境热力学与LPSC_LLZO谱学路线.md",
        "slug": "zhou2025",
        "filename": "2026-05-16-Zhou2025-非环境热力学与LPSC-LLZO谱学路线.md",
        "date": "2026-05-16",
        "title": "固态电池里“被忽略的能量项”：从 Zhou 2025 看 LPSC/LLZO 的谱学研究新机会",
        "subtitle": "复合正极局部应力可达数百 MPa 时，机械场不再只是背景，而是直接参与反应驱动力分配",
        "tags": ["固态电池", "谱学", "热力学", "LPSC", "LLZO"],
        "image_dirs": [
            (
                WECHAT
                / "电池领域思路"
                / "Zhou2025-非环境热力学与LPSC_LLZO谱学路线"
                / "zhou2025_imgs",
                None,
            ),
        ],
        "replacements": [
            ("./zhou2025_imgs/", "/img/posts/zhou2025/"),
            ("zhou2025_imgs/", "/img/posts/zhou2025/"),
        ],
    },
    {
        "src": WECHAT / "屏蔽材料开发介绍" / "Koker文章介绍" / "2026koker-pft-wechat.md",
        "slug": "koker-pft",
        "filename": "2026-05-14-Koker2026-PFT声子微调.md",
        "date": "2026-05-14",
        "title": "速览笔记：PFT 如何让 MLIP 真正「听懂」声子？",
        "subtitle": "在能量/力/应力之外直接监督 Hessian，声子热力学误差平均约降 55%",
        "tags": ["MLIP", "声子", "PFT", "势函数"],
        "image_dirs": [
            (WECHAT / "屏蔽材料开发介绍" / "Koker文章介绍" / "images", None),
        ],
        "replacements": [(r"images/", "/img/posts/koker-pft/")],
    },
    {
        "src": WECHAT / "屏蔽材料开发介绍" / "Liu文章FIRE介绍" / "2026liu-fire-wechat.md",
        "slug": "liu-fire",
        "filename": "2026-05-15-Liu2026-FIRE固固界面精调.md",
        "date": "2026-05-15",
        "title": "一文速览：FIRE 如何把通用 MLIP「拉」到固固界面精度？",
        "subtitle": "FIRE 不是新势函数，而是「高效采样 + Replay 精调」框架，在六类电池界面体系上显著降误差",
        "tags": ["MLIP", "界面", "FIRE", "固态电池"],
        "image_dirs": [
            (WECHAT / "屏蔽材料开发介绍" / "Liu文章FIRE介绍" / "图片", None),
        ],
        "replacements": [(r"图片/", "/img/posts/liu-fire/")],
    },
    {
        "src": WECHAT / "Kim-文章介绍" / "2026kim-polarizable-multipoles-wechat.md",
        "slug": "kim-polar",
        "filename": "2026-05-17-Kim2026-可极化多极矩长程电静学.md",
        "date": "2026-05-17",
        "title": "一篇讲透：为什么“可极化多极矩”能让材料力场更懂电学？",
        "subtitle": "局域可学习多极矩 + 非自洽诱导响应 + Ewald，系统提升力场与电响应预测",
        "tags": ["MLIP", "极化", "长程电静学", "势函数"],
        "image_dirs": [
            (WECHAT / "Kim-文章介绍" / "imgs" / "paper", "paper"),
        ],
        "replacements": [(r"imgs/paper/", "/img/posts/kim-polar/paper/")],
    },
]


def split_front_matter(text: str):
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :].lstrip("\n"), text[4:end]
    return text, None


def strip_duplicate_h1(body: str, title: str) -> str:
    lines = body.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].startswith("# "):
        h1 = lines[i][2:].strip()
        if h1 == title.strip() or h1.replace('"', "“") == title.replace('"', "“"):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            body = "\n".join(lines[i:])
    return body.lstrip("\n")


def apply_replacements(body: str, replacements):
    for old, new in replacements:
        body = body.replace(old, new)
    return body


def build_front_matter(cfg):
    tags_yaml = "\n".join(f"    - {t}" for t in cfg["tags"])
    return (
        "---\n"
        "layout:     post\n"
        f"title:      {cfg['title']}\n"
        f"subtitle:   {cfg['subtitle']}\n"
        f"date:       {cfg['date']}\n"
        "author:     qqz\n"
        "header-img: img/post-bg-desk.jpg\n"
        "catalog:    true\n"
        "tags:\n"
        f"{tags_yaml}\n"
        "---\n"
    )


def main():
    old = list(POSTS.glob("*.md"))
    for f in old:
        f.unlink()
    print(f"Deleted {len(old)} old posts")

    for cfg in POSTS_CONFIG:
        if not cfg["src"].exists():
            raise FileNotFoundError(cfg["src"])

        dest_img = IMG / cfg["slug"]
        if dest_img.exists():
            shutil.rmtree(dest_img)
        dest_img.mkdir(parents=True, exist_ok=True)

        for src_dir, sub in cfg["image_dirs"]:
            if not src_dir.exists():
                print(f"WARN missing image dir: {src_dir}")
                continue
            target = dest_img / sub if sub else dest_img
            target.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, target / f.name)

        raw = cfg["src"].read_text(encoding="utf-8")
        body, _ = split_front_matter(raw)
        body = strip_duplicate_h1(body, cfg["title"])
        body = apply_replacements(body, cfg["replacements"])

        out = POSTS / cfg["filename"]
        out.write_text(
            build_front_matter(cfg) + "\n" + body + ("\n" if not body.endswith("\n") else ""),
            encoding="utf-8",
        )
        print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
