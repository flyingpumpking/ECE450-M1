import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# 加载中文预训练 BERT 模型
MODEL_NAME = "google-bert/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)  # 👈 确保用 BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def get_bert_embedding(text):
    """生成文本的 BERT 语义嵌入"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用 [CLS] token 的嵌入作为句子表示
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def dynamic_threshold(text, task_type="union"):
    """
    动态阈值，根据任务长度调整：
    - 并集任务 (union)：任务越长，允许的相似度越低
    - 交集任务 (intersection)：任务越长，允许的相似度越低
    """
    # 基础阈值
    if task_type == "union":
        base_threshold = 0.85
    elif task_type == "intersection":
        base_threshold = 0.35
    else:
        raise ValueError("task_type must be 'union' or 'intersection'")

    # 任务越长，允许的相似度越低（可以线性或指数调节）
    # 用 token 数量衡量任务长度
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=128)
    token_length = tokens.shape[1]

    # 指数衰减函数，长度越长，阈值越低
    decay_factor = 0.95
    threshold = base_threshold * (decay_factor ** (token_length / 10))
    return max(0.1, threshold)  # 避免阈值过低


def verify_task_split(C, A, B, verbose=False):
    """
    验证任务拆分的正确性：
    1. A + B 是否语义接近 C（并集验证）
    2. A 和 B 是否语义不重叠（交集验证）
    """
    # 一、并集验证
    combined = A + " " + B
    emb_C = get_bert_embedding(C)
    emb_combined = get_bert_embedding(combined)

    union_sim = cosine_similarity(emb_C, emb_combined)[0][0]
    union_threshold = dynamic_threshold(C, "union")

    # 二、交集验证
    emb_A = get_bert_embedding(A)
    emb_B = get_bert_embedding(B)

    intersection_sim = cosine_similarity(emb_A, emb_B)[0][0]
    intersection_threshold = dynamic_threshold(C, "intersection")

    # 三、决策逻辑
    is_union_valid = union_sim >= union_threshold
    is_intersection_valid = intersection_sim <= intersection_threshold

    if verbose:
        print(f"C: '{C}'")
        print(f"A: '{A}'")
        print(f"B: '{B}'")
        print(f"combined: '{combined}'")
        print(f"Union 语义相似度: {union_sim:.4f}, 阈值: {union_threshold:.4f}")
        print(f"Intersection 语义相似度: {intersection_sim:.4f}, 阈值: {intersection_threshold:.4f}")
        print(f"并集验证通过: {is_union_valid}, 交集验证通过: {is_intersection_valid}")

    return is_union_valid and is_intersection_valid

# 示例任务
A = "Use a power screwdriver to remove the refrigerator back panel/top cover, separate the plastic shell from the metal frame, and sort the screws."
B = "Clip structure for manual separation of housing from metal frame."
C = "Use a power screwdriver to remove backplate/top cover screws in bulk."
result = verify_task_split(A, B, C, verbose=True)
print("任务拆分是否符合语义正确性要求:", result)