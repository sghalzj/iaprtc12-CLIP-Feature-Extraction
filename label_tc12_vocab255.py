import os, re
from collections import Counter, defaultdict
import numpy as np
import scipy.io as sio
import xml.etree.ElementTree as ET

# === 配置参数 ===
ANN_DIR = r"S:\dataset\iaprtc12\unpack\iaprtc12\annotations_complete_eng"
OUT_DIR = r"S:\SCH-main"
USE_STOPWORDS = True
MIN_DF = 5
TOP_K = 255
EXT_FILTER = {'.eng'}

DEFAULT_STOPWORDS = {
    # 冠词 (Articles)
    "a", "an", "the",

    # 最常用代词 (Pronouns)
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",

    # 指示词和不定代词 (Demonstratives & Indefinite Pronouns)
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose",
    "other", "another", "same", "such",

    # 介词 (Prepositions)
    "about", "above", "across", "after", "against", "along", "among", "around", "at", "before",
    "behind", "below", "beneath", "beside", "between", "beyond", "by", "down", "during", "for",
    "from", "in", "inside", "into", "near", "of", "off", "on", "onto", "out", "outside", "over",
    "through", "throughout", "to", "toward", "under", "until", "up", "upon", "with", "within", "without",

    # 连词 (Conjunctions)
    "and", "or", "but", "nor", "so", "yet",
    "if", "because", "although", "since", "unless", "while", "where", "when", "as",

    # 动词 (Verbs)
    "is", "are", "was", "were", "be", "being", "been",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",

    # 副词 (Adverbs)
    "very", "really", "quite", "too", "just", "only", "also", "well", "much", "more", "most",
    "even", "still", "almost", "enough", "however", "therefore", "thus", "hence",
    "now", "then", "here", "there", "when", "where", "why", "how",

    # 限定词和数量词 (Determiners & Quantifiers)
    "all", "any", "each", "every", "no", "none", "some", "lot", "few", "many", "several", "both", "either", "neither",
    "own", "same", "so", "than", "too",

    # 感叹词和其他 (Interjections & Others)
    "oh", "yes", "no", "not", "ain", "aren", "couldn", "didn", "doesn", "don", "hadn", "hasn",
    "haven", "isn", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",

    # 常见词汇
    "get", "got", "go", "goes", "going", "went", "come", "came", "see", "saw", "say", "said",
    "know", "knew", "think", "thought", "take", "took", "make", "made", "use", "used",
    "team",  # 保留，因为不在词语列表中
    "brown", "blue", "yellow", "pink",  # 保留这些颜色，因为它们不在词语列表中
    "next", "behind", "front", "left", "right",  # 保留这些方位词
    "high", "low", "broad", "narrow",
    "up", "down", "out", "off", "around", "some", "many", "few",
    "several", "both", "large", "small", "long", "short", "steep",
    "flat", "little", "big",  # 移除了重复的 "high", "low"
    "flowers", "leaves",  # 保留，使用复数形式

    # 数字单词
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",

    # 动作状态词（移除了在词语列表中出现的）
    "posing", "walking", "lying", "dancing", "eating",
    "surrounded", "covered", "illuminated", "burning",

    # 移除了以下词语（因为它们存在于词语列表中）：
    # "back", "bench", "chair", "green", "grey", "orange", "red", "white", "table",
    # "rock", "people", "person", "man", "woman", "group", "life", "light", "power",
    # "side", "playing", "racing", "riding", "cycling", "holding", "standing", "sitting",
    # "middle", "rise" (这些也在之前的停用词中但已移除)
}



def norm_token(s: str) -> list:
    s = s.lower()
    # 更精确的文本清理
    s = re.sub(r"[^\w\s]", " ", s)  # 保留字母数字和空格
    s = re.sub(r"\d+", " ", s)  # 去除数字
    s = re.sub(r"\s+", " ", s)  # 合并多个空格
    s = s.strip()

    tokens = s.split()
    if USE_STOPWORDS:
        tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS and len(t) > 2]  # 额外过滤短词
    return tokens

def read_text(fp, encs=("utf-8", "latin1", "cp1252")):
    for enc in encs:
        try:
            with open(fp, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            pass
    with open(fp, "rb") as f:
        return f.read().decode("latin1", "ignore")

def read_text(fp, encs=("utf-8", "latin1", "cp1252")):
    for enc in encs:
        try:
            with open(fp, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            pass
    with open(fp, "rb") as f:
        return f.read().decode("latin1", "ignore")

def collect_title_description_tokens(ann_dir):
    per_image_tokens = {}
    total_files = 0
    for root, _, files in os.walk(ann_dir):
        subdir = os.path.basename(root)
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in EXT_FILTER:
                continue
            base = os.path.splitext(file)[0]
            key = f"{subdir}/{base}"
            full_path = os.path.join(root, file)
            raw = read_text(full_path)

            try:
                tree = ET.fromstring(raw)
                # 只用 DESCRIPTION，不再拼接 TITLE
                desc = tree.findtext("DESCRIPTION", "")
                full_text = desc.strip()
                tokens = norm_token(full_text)
                if tokens:
                    per_image_tokens[key] = tokens
                    total_files += 1
            except ET.ParseError:
                print(f"⚠️ 无法解析：{full_path}")
    print(f"共读取 .eng 文件: {total_files} 个（有效样本）")
    return per_image_tokens

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"读取注释目录：{ANN_DIR}")
    annotations = collect_title_description_tokens(ANN_DIR)
    keys = sorted(annotations.keys(), key=lambda x: [int(t) if t.isdigit() else t for t in re.findall(r'\d+|\D+', x)])
    N = len(keys)
    if N == 0:
        raise RuntimeError("❌ 没有读取到任何有效标签。请确认文件结构和内容格式。")

    # 统计词频
    freq = Counter(t for tags in annotations.values() for t in tags)
    if MIN_DF > 0:
        freq = Counter({w: c for w, c in freq.items() if c >= MIN_DF})
        if not freq:
            raise RuntimeError("⚠️ MIN_DF 太高，词表为空，请调整参数。")

    vocab = [w for w, _ in freq.most_common(TOP_K)]
    K = len(vocab)
    if K < TOP_K:
        print(f"[提示] 实际词表大小仅为 {K}，不足 {TOP_K}。")

    word2id = {w: i for i, w in enumerate(vocab)}

    # 生成多热标签
    Y = np.zeros((N, K), dtype=np.uint8)
    for i, k in enumerate(keys):
        for t in annotations[k]:
            j = word2id.get(t)
            if j is not None:
                Y[i, j] = 1

    # 保存
    vocab_path = os.path.join(OUT_DIR, "iaprtc12_vocab255.txt")
    mat_path   = os.path.join(OUT_DIR, "iaprtc12_labels255.mat")
    npz_path   = os.path.join(OUT_DIR, "iaprtc12_labels255.npz")

    np.savetxt(vocab_path, np.array(vocab, dtype=str), fmt="%s", encoding="utf-8")
    sio.savemat(mat_path, {
        "keys":     np.array(keys, dtype=object),
        "vocab255": np.array(vocab, dtype=object),
        "Y_multi":  Y
    }, do_compression=True)
    np.savez_compressed(npz_path,
        keys=np.array(keys, dtype=object),
        vocab255=np.array(vocab, dtype=object),
        Y_multi=Y
    )

    density = float(Y.sum()) / (N * K)
    print("✅ 标签生成完成")
    print(f"样本数 N = {N}, 标签维度 K = {K}, 标签密度 = {density:.6f}")
    print("输出文件：")
    print(" -", vocab_path)
    print(" -", mat_path)
    print(" -", npz_path)
    print("📎 对齐说明：图像路径为 images/{keys[i]}.jpg")

if __name__ == "__main__":
    main()
