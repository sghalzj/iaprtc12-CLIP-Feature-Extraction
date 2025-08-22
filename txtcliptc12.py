import os
import re
import numpy as np
import torch
import clip
from tqdm import tqdm
import scipy.io as sio
import xml.etree.ElementTree as ET

# ===== 手动配置 =====
ANN_FULL_DIR = r"S:\dataset\IAPR-TC-12\captions"
OUT_DIR = r"S:\SCH-main"
MODEL_NAME = "ViT-B/16"
BATCH_SIZE = 1
NORMALIZE = False


# ====================

def read_text(fp, encs=("utf-8", "latin1", "cp1252")):
    for enc in encs:
        try:
            with open(fp, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            pass
    with open(fp, "rb") as f:
        return f.read().decode("latin1", "ignore")


def split_into_sentences(text):
    sentences = re.split(r'[.?!；;。]+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_description_from_xml(raw):
    """从XML内容中提取描述文本"""
    try:
        tree = ET.fromstring(raw)
        desc_elem = tree.find("DESCRIPTION")
        if desc_elem is not None and desc_elem.text:
            return desc_elem.text.strip()
    except ET.ParseError:
        # XML解析失败，尝试正则表达式
        desc_match = re.search(r'<DESCRIPTION[^>]*>(.*?)</DESCRIPTION>', raw, re.IGNORECASE | re.DOTALL)
        if desc_match:
            desc_text = desc_match.group(1).strip()
            desc_text = re.sub(r'<[^>]+>', '', desc_text)
            return desc_text
    return ""


def process_all_annotation_files(ann_dir):
    """处理所有注释文件并按顺序返回文本和keys"""
    all_texts = []
    all_keys = []

    # 获取所有 .eng 文件并按文件名排序
    eng_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.eng')])

    for file in tqdm(eng_files, desc="Processing annotation files"):
        full_path = os.path.join(ann_dir, file)
        raw = read_text(full_path)

        # 提取描述文本
        desc_text = extract_description_from_xml(raw)

        # 清理文本
        if desc_text:
            desc_text = re.sub(r'\s+', ' ', desc_text).strip()
        else:
            desc_text = "a photo of something."  # 默认描述

        # 使用文件名作为key（去掉扩展名）
        key = os.path.splitext(file)[0]

        all_texts.append(desc_text)
        all_keys.append(key)

    print(f"成功处理 {len(all_texts)} 个文件")
    return all_keys, all_texts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 处理所有注释文件
    print("Processing all annotation files...")
    keys, texts = process_all_annotation_files(ANN_FULL_DIR)

    # 显示一些样本
    print("\nSample extracted texts:")
    for i, (key, text) in enumerate(zip(keys[:5], texts[:5])):
        print(f"{i + 1}. {key}: '{text[:100]}...'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP {MODEL_NAME} on {device} ...")
    model, _ = clip.load(MODEL_NAME, device=device)
    model.eval()

    feats = np.zeros((len(keys), 512), dtype=np.float32)

    for i, t in enumerate(tqdm(texts, desc="Encoding with sentence avg")):
        sentences = split_into_sentences(t)
        if len(sentences) == 0:
            sentences = ["a photo."]

        sub_feats = []
        for sent in sentences:
            tok = clip.tokenize([sent], truncate=True).to(device)
            with torch.no_grad():
                f = model.encode_text(tok)
            f = f.float().cpu().numpy()
            sub_feats.append(f[0])

        feat_avg = np.mean(sub_feats, axis=0)
        if NORMALIZE:
            norm = np.linalg.norm(feat_avg) + 1e-12
            feat_avg = feat_avg / norm

        feats[i] = feat_avg

    # 保存
    out_npz = os.path.join(OUT_DIR, "clip_vitb16_text_embeds_sentence_avg.npz")
    out_mat = os.path.join(OUT_DIR, "clip_vitb16_text_embeds_sentence_avg.mat")
    np.savez_compressed(out_npz, keys=np.array(keys), feats=feats, texts=np.array(texts))
    sio.savemat(out_mat, {"keys": keys, "feats": feats, "texts": texts}, do_compression=True)

    print(f"Saved:\n  {out_npz}\n  {out_mat}")
    print(f"Total files processed: {len(keys)}")


if __name__ == "__main__":
    main()
