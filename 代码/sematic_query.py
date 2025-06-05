import redis
import numpy as np
import os
import pandas as pd
import re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer

r = redis.Redis(host="192.168.0.207", port=6380)

# ==================== 修改此处加载你要测试的模型 ====================
# model_name = "BCE"
# model = SentenceTransformer("maidalun1020/bce-embedding-base_v1")

# model_name = "KaLM"
# model = SentenceTransformer("HIT-TMG/KaLM-embedding-multilingual-mini-v1")

model_name = "Linq"
model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral")

# model_name = "BGE"
# model = SentenceTransformer("BAAI/bge-base-zh")
# =====================================================================

print(f"✅ 模型 {model_name} 加载完成，输入查询语句（纯文本），输入 'exit' 退出，或输入 'batch' 批量查询。")

def run_query(query_text, top_k=3):
    """
    执行单条查询并返回结果列表（含 score）
    """
    query_vec = model.encode(query_text, normalize_embeddings=True)
    query_bytes = np.array(query_vec, dtype=np.float32).tobytes()

    try:
        res = r.execute_command(
            "FT.SEARCH",
            "doc_idx",
            f"*=>[KNN {top_k} @vec $vec AS score]",
            "PARAMS", "2", "vec", query_bytes,
            "DIALECT", "2"
        )
    except redis.exceptions.ResponseError as e:
        print(f" 查询失败：{str(e)}")
        return []

    results = []
    if len(res) <= 1:
        return results

    for i in range(1, len(res), 2):
        doc_id = res[i].decode()
        fields_list = res[i + 1]
        score_value = None
        text = "[无 text 字段]"
        for k, v in zip(fields_list[::2], fields_list[1::2]):
            key = k.decode() if isinstance(k, bytes) else str(k)
            if key == "score":
                score_value = float(v)
            elif key == "text":
                text = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        results.append({
            "doc_id": doc_id,
            "score": score_value,
            "text": text
        })
    return results

def batch_query_from_file(file_path="data/Ques_CA.txt", output_path=None, top_k=10):
    """
    批量读取纯文本查询（按每行一个查询），并将结果输出为表格（含 hit@1, hit@3, hit@10）
    gold_doc 根据行号自动赋值：第一行对应 doc:1，第二行对应 doc:2……
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件未找到：{file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]

    records = []
    for idx, query in enumerate(queries):
        gold_doc = f"doc:{idx+1}"
        res = run_query(query, top_k=top_k)
        doc_ids = [item["doc_id"] for item in res]

        hit1  = int(doc_ids and doc_ids[0] == gold_doc)
        hit3  = int(gold_doc in doc_ids[:3])
        hit10 = int(gold_doc in doc_ids[:10])

        records.append({
            "原始查询":  query,
            "gold_doc":  gold_doc,
            "top1_doc":  doc_ids[0] if doc_ids else "",
            "top3_docs": ", ".join(doc_ids[:3]),
            "top10_docs": ", ".join(doc_ids),
            "hit@1":     hit1,
            "hit@3":     hit3,
            "hit@10":    hit10
        })

    if not records:
        print("⚠️ 无有效查询，未生成结果。")
        return

    # 构造 DataFrame
    df = pd.DataFrame(records)

    # 计算整体命中率
    total_q = len(df)
    hit1_rate  = df["hit@1"].sum()  / total_q
    hit3_rate  = df["hit@3"].sum()  / total_q
    hit10_rate = df["hit@10"].sum() / total_q

    # 打印整体命中率
    print(f"📊 共 {total_q} 条查询，整体命中率：")
    print(f"  • Hit@1 : {hit1_rate:.2%} ({df['hit@1'].sum()}/{total_q})")
    print(f"  • Hit@3 : {hit3_rate:.2%} ({df['hit@3'].sum()}/{total_q})")
    print(f"  • Hit@10: {hit10_rate:.2%} ({df['hit@10'].sum()}/{total_q})")

    # 保存到 Excel
    if not output_path:
        output_path = f"评估结果_{model_name}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"✅ 批量查询完成，结果已保存到：{output_path}")


# === 主交互逻辑 ===
while True:
    line = input("\n请输入查询语句：")
    if line.lower() in ("exit", "quit"):
        print(" 已退出查询系统。")
        break
    if line.lower() == "batch":
        batch_query_from_file()
        continue

    # 单条查询：直接将输入的纯文本送入 run_query
    results = run_query(line)
    if not results:
        print("😕 未找到匹配结果。")
        continue

    print("\n 查询结果：")
    for item in results:
        print(f"{item['doc_id']} (score={item['score']:.4f}): {item['text'][:80]}...")
