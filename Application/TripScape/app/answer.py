import sqlite3
import math
import re
from typing import List, Dict, Tuple, Optional

import jieba
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter

# ---------- 配置区 ----------
DB_PATH = "data/landmark.db"
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
TOP_K = 3                # 返回的相关结果数
ENABLED_ONLY = True      # 是否只检索已启用（enabled=1）的数据
VECTOR_WEIGHT = 0.6      # 混合检索时向量的权重（剩余为 BM25 权重）
SCORE_THRESHOLD = 0.2    # 最低置信度阈值，低于此值将返回兜底回复


# ---------- BM25 关键词检索算法 ----------
class BM25:
    """
    手写 BM25 实现，用于关键词匹配，弥补向量检索对专有名词不够敏感的问题。
    """

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_lengths = [len(doc) for doc in documents]
        self.avg_length = sum(self.doc_lengths) / len(documents)
        self.corpus_size = len(documents)

        # 分词
        self.tokenized_docs = [list(jieba.cut(doc)) for doc in documents]

        # 计算 IDF
        doc_count_per_token = Counter()
        for tokens in self.tokenized_docs:
            doc_count_per_token.update(set(tokens))

        self.idf = {}
        for token, doc_freq in doc_count_per_token.items():
            self.idf[token] = math.log(
                (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1
            )

    def get_score(self, query: str, doc_idx: int) -> float:
        query_tokens = list(jieba.cut(query))
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        doc_freq = Counter(doc_tokens)

        score = 0.0
        for token in query_tokens:
            if token not in self.idf:
                continue
            tf = doc_freq.get(token, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_length))
            score += self.idf[token] * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        scores = [(idx, self.get_score(query, idx)) for idx in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------- 数据加载 ----------
def load_documents(db_path: str, enabled_only: bool = True) -> List[Dict[str, str]]:
    """
    从 SQLite 数据库的 knowledge_base 表加载数据，
    动态适应不同的列组合（可能存在的列：id, question, keywords, answer, enabled, created_at 等）。
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_base'")
    if not cursor.fetchone():
        raise ValueError("数据库中没有 'knowledge_base' 表，请检查数据库文件。")

    # 2. 获取表中所有列名
    cursor.execute("PRAGMA table_info(knowledge_base)")
    columns_info = cursor.fetchall()
    all_columns = [col[1] for col in columns_info]

    # 3. 确定必须存在的列（id, question, keywords, answer）
    required = ['id', 'question', 'keywords', 'answer']
    missing = [col for col in required if col not in all_columns]
    if missing:
        raise ValueError(f"数据库表缺少必要列：{missing}，请检查表结构。")

    # 4. 构建查询列列表（全部存在的列，但至少包含必须列）
    select_columns = [col for col in all_columns if col in required or col in ['enabled', 'created_at']]
    # 如果 enabled 和 created_at 不存在，也没关系

    # 5. 构建 SELECT 语句
    sql = f"SELECT {', '.join(select_columns)} FROM knowledge_base"

    # 如果存在 enabled 列且 enabled_only 为 True，添加 WHERE 条件
    if 'enabled' in all_columns and enabled_only:
        sql += " WHERE enabled = 1"

    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()

    # 6. 处理每一行，构造文档字典
    documents = []
    for row in rows:
        # 将行数据映射到字典（列名 -> 值）
        row_dict = dict(zip(select_columns, row))
        doc_id = row_dict['id']
        question = row_dict.get('question') or ""
        keywords = row_dict.get('keywords') or ""
        answer = row_dict.get('answer') or ""

        # 构建用于检索的全文
        full_text = f"问题：{question}\n关键词：{keywords}\n答案：{answer}"
        if keywords:
            full_text += f"\n{keywords}"

        documents.append({
            "id": doc_id,
            "question": question,
            "keywords": keywords,
            "answer": answer,
            "full_text": full_text,
        })

    print(f"✅ 从数据库加载 {len(documents)} 条记录（列：{select_columns}）")
    return documents
# ---------- 向量索引构建 ----------
def build_vector_index(documents: List[Dict[str, str]]) -> Tuple[faiss.Index, SentenceTransformer]:
    """将文档全文编码为向量，构建 FAISS 内积索引（归一化后等价余弦相似度）"""
    print("⏳ 正在加载向量模型...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    texts = [doc["full_text"] for doc in documents]
    print("⏳ 正在生成文档向量...")
    embeddings = encoder.encode(texts, convert_to_numpy=True, batch_size=32)
    # 归一化，使用内积作为相似度
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"✅ 向量索引构建完成，共 {index.ntotal} 条")
    return index, encoder
# ---------- 混合检索 ----------
def hybrid_search(
    query: str,
    documents: List[Dict[str, str]],
    vector_index: faiss.Index,
    encoder: SentenceTransformer,
    bm25_model: BM25,
    top_k: int = TOP_K,
    vector_weight: float = VECTOR_WEIGHT,
) -> List[Dict]:
    """
    向量检索（语义）与 BM25（关键词）混合加权，返回 Top-K 结果。
    """
    # 1. 向量检索
    query_vec = encoder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    vec_scores, vec_indices = vector_index.search(query_vec, top_k * 2)  # 多取候选
    vec_scores = vec_scores[0]
    vec_indices = vec_indices[0]

    # 2. BM25 检索
    bm25_candidates = bm25_model.search(query, top_k * 2)

    # 3. 收集所有候选索引
    vec_scores_dict = {idx: float(score) for idx, score in zip(vec_indices, vec_scores)}
    bm25_scores_dict = {idx: score for idx, score in bm25_candidates if score > 0}
    all_indices = set(vec_scores_dict.keys()) | set(bm25_scores_dict.keys())

    if not all_indices:
        return []

    # 4. 归一化（Min-Max）
    def min_max_normalize(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        min_v, max_v = min(scores.values()), max(scores.values())
        if max_v == min_v:
            return {k: 1.0 for k in scores}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

    norm_vec = min_max_normalize(vec_scores_dict)
    norm_bm25 = min_max_normalize(bm25_scores_dict)

    # 5. 加权融合
    final_scores = {}
    for idx in all_indices:
        final_scores[idx] = (
            vector_weight * norm_vec.get(idx, 0.0)
            + (1 - vector_weight) * norm_bm25.get(idx, 0.0)
        )

    # 6. 排序并取 Top-K
    sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in sorted_items:
        results.append({
            "document": documents[idx],
            "score": score,
            "detail": {
                "vec_score": norm_vec.get(idx, 0.0),
                "bm25_score": norm_bm25.get(idx, 0.0),
            },
        })
    return results


# ---------- 结果格式化（单问题） ----------
def format_answer(query: str, retrieved_results: List[Dict], threshold: float = 0.2) -> str:
    """
    将检索结果格式化为用户友好的回答。
    参数：
        query: 用户原始问题
        retrieved_results: 检索结果列表，每个元素包含 'document' 和 'score'
        threshold: 置信度阈值，低于此值视为无匹配
    返回：
        格式化后的回答字符串
    """
    # 1. 检查是否有有效结果
    if not retrieved_results or retrieved_results[0]["score"] < threshold:
        return "🤔 抱歉，我的本地知识库中暂时没有找到与您问题匹配的信息。请尝试更具体的关键词。"

    # 2. 纯地名查询优先展示“介绍”类记录
    question_words = ["多少", "哪里", "什么", "怎么", "如何", "哪", "几", "多", "吗", "呢", "吧"]
    if not any(word in query for word in question_words):
        for res in retrieved_results:
            q = res["document"].get("question", "")
            if "介绍" in q:
                retrieved_results.remove(res)
                retrieved_results.insert(0, res)
                print("💡 检测到纯地名查询，优先展示介绍信息")
                break

    # 3. 提取最佳结果
    best = retrieved_results[0]
    doc = best["document"]

    answer_text = doc.get("answer", "（该记录暂无详细答案）")
    question_text = doc.get("question", "（无对应问题）")
    keywords_text = doc.get("keywords", "（无关键词）")

    # 4. 构建输出
    result = f"🏞️ {answer_text}\n\n"
    result += f"📌 匹配到的问题：{question_text}\n"
    result += f"🔑 关键词：{keywords_text}"

    # 5. 推荐其他相关问题
    if len(retrieved_results) > 1:
        other_questions = [
            r["document"].get("question", "")
            for r in retrieved_results[1:]
            if r["document"].get("question")
        ]
        if other_questions:
            result += f"\n\n💡 您可能还感兴趣：{'、'.join(other_questions)}"

    # 6. 置信度
    result += f"\n\n📊 [检索置信度: {best['score']:.2%}]"
    return result


# ---------- 主接口 ----------
class AIGuideRAG:
    """本地算法 AI 导游，支持单问题和复合问题（自动拼接）"""

    def __init__(
        self,
        db_path: str = DB_PATH,
        enabled_only: bool = ENABLED_ONLY,
        top_k: int = TOP_K,
        vector_weight: float = VECTOR_WEIGHT,
        score_threshold: float = SCORE_THRESHOLD,
    ):
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.score_threshold = score_threshold

        print("🚀 AI 导游初始化中...")

        self.documents = load_documents(db_path, enabled_only)
        if not self.documents:
            raise ValueError("没有加载到任何有效数据，请检查数据库。")

        full_texts = [doc["full_text"] for doc in self.documents]
        self.bm25 = BM25(full_texts)
        self.vector_index, self.encoder = build_vector_index(self.documents)

        print("🎉 初始化完成")

    # ==================== 单问题检索 ====================
    def _single_ask(self, query: str) -> str:
        """执行单问题检索并返回完整格式化回答"""
        results = hybrid_search(
            query=query,
            documents=self.documents,
            vector_index=self.vector_index,
            encoder=self.encoder,
            bm25_model=self.bm25,
            top_k=self.top_k,
            vector_weight=self.vector_weight,
        )
        return format_answer(query, results, self.score_threshold)

    # ==================== 切分子问题 ====================
    def _split_questions(self, query: str) -> List[str]:
        """
        将复合问题拆分为独立子句。
        - 按中文句号、问号、感叹号、分号切分
        - 保留所有长度>1且非纯标点的片段
        - 若片段不含疑问词且长度较长（>10），则合并到前一句（避免误切描述句）
        """
        # 1. 按标点切分
        parts = re.split(r'([。？?！!；;])', query)
        sentences = []
        buffer = ""
        for part in parts:
            if part in "。？?！!；;":
                buffer += part
                if buffer.strip():
                    sentences.append(buffer.strip())
                buffer = ""
            else:
                buffer += part
        if buffer.strip():
            sentences.append(buffer.strip())

        # 2. 过滤与合并
        question_markers = ["吗", "呢", "怎么", "如何", "什么", "哪", "几", "多", "多少",
                            "哪里", "何时", "为啥", "为什么", "是否", "有", "干啥", "干嘛", "啥"]
        noun_questions = ["用途", "门票", "地址", "历史", "简介", "开放时间", "交通", "怎么去"]

        valid = []
        for s in sentences:
            if len(s) < 2:
                continue
            is_question = (
                any(m in s for m in question_markers) or
                len(s) <= 6 or
                any(n in s for n in noun_questions) or
                s.endswith(("？", "?"))
            )
            if is_question:
                valid.append(s)
            else:
                if valid:
                    valid[-1] += s
                else:
                    valid.append(s)
        return valid

    # ==================== 提取景点名 ====================
    def _extract_scenic_name(self, question: str) -> Optional[str]:
        """
        从问题中提取景点名。
        匹配模式：景点名 + 分隔词（位于、是、介绍、门票、高度、用途等）
        """
        separators = (
            "位于|在|是|介绍|门票|高度|长度|面积|历史|用途|做什么|干什么"
            "|怎么|如何|多长|多高|多少|什么时候|建于|为什么|什么"
            "|在哪|在哪里|有哪些|是谁"
        )
        # 从问题开头匹配到第一个分隔词
        match = re.match(rf'^([^，,。.、？?！!；;]+?)(?:{separators})', question)
        if match:
            return match.group(1).strip()
        # 如果匹配不到，则取第一个连续中文/英文字符串
        match = re.match(r'^([\u4e00-\u9fa5a-zA-Z]+)', question)
        if match:
            return match.group(1).strip()
        return None

    # ==================== 补全子问题 ====================
    def _complete_sub_question(self, sub_q: str, scenic_name: str) -> str:
        """
        为缺少景点名的子问题补全景点名。
        例如："用途" -> "天坛用途"
        """
        # 如果景点名为空，或子问题已经包含景点名，不做修改
        if not scenic_name or scenic_name in sub_q:
            return sub_q
        # 如果子问题以"的"开头，直接拼接（如"的用途"）
        if sub_q.startswith("的"):
            return f"{scenic_name}{sub_q}"
        # 否则加上景点名（不加"的"，避免冗余）
        return f"{scenic_name}{sub_q}"

    # ==================== 合并多个答案 ====================
    def _merge_answers(self, sub_questions: List[str], raw_answers: List[str], scenic_name: str) -> str:
        """
        将多个答案合并成一段连贯的自然语言。
        - 去掉重复的景点名前缀
        - 根据答案类型选择合适的连接词
        - 统一标点符号（确保不重复、不缺失）
        """
        if not raw_answers:
            return ""

        # 如果只有一个答案，直接返回（但需要确保标点）
        if len(raw_answers) == 1:
            ans = raw_answers[0]
            return ans if ans.endswith(("。", "！", "？")) else ans + "。"

        # ---- 1. 去掉后续答案中的景点名前缀 ----
        if scenic_name:
            for i in range(1, len(raw_answers)):
                ans = raw_answers[i]
                if ans.startswith(scenic_name):
                    raw_answers[i] = ans[len(scenic_name):].lstrip("，,、的")
                else:
                    patterns = [
                        rf'^{scenic_name}是',
                        rf'^{scenic_name}位于',
                        rf'^{scenic_name}在',
                        rf'^{scenic_name}的',
                    ]
                    for pat in patterns:
                        match_remove = re.match(pat, ans)
                        if match_remove:
                            raw_answers[i] = ans[match_remove.end():].lstrip("，,、")
                            break

        # ---- 2. 去除第一个答案末尾的句号、感叹号、问号（避免重复标点） ----
        raw_answers[0] = re.sub(r'[。！？！!?]+$', '', raw_answers[0]).strip()

        # ---- 3. 根据答案数量选择连接方式 ----
        merged = ""
        if len(raw_answers) == 2:
            # 两个答案：判断是否需要特殊连接
            if "位于" in raw_answers[0] or "在" in raw_answers[0]:
                if "用途" in sub_questions[1] or "做什么" in sub_questions[1] or "干啥" in sub_questions[1]:
                    merged = f"{raw_answers[0]}，主要用于{raw_answers[1]}"
                else:
                    merged = f"{raw_answers[0]}，{raw_answers[1]}"
            else:
                merged = f"{raw_answers[0]}，{raw_answers[1]}"
        else:
            # 三个及以上：用分号连接，最后一个用"以及"
            # 去除每个子答案末尾的标点（可选，但为了美观，我们只处理最后统一加）
            # 这里保持原有逻辑，但分号后不加额外标点
            merged = "；".join(raw_answers[:-1]) + f"；以及{raw_answers[-1]}"

        # ---- 4. 加上景点名前缀（如果还没加且需要） ----
        if scenic_name and not merged.startswith(scenic_name):
            if not re.match(r'^(位于|在|是|有|由)', merged):
                merged = f"{scenic_name}{merged}"

        # ---- 5. 统一标点：确保结尾有句号，且内部没有重复标点 ----
        # 去掉内部可能残留的句号后接逗号的情况
        merged = re.sub(r'[。！？]+\s*，', '，', merged)  # 如“。，” -> “，”
        merged = re.sub(r'，+', '，', merged)  # 多个逗号合并
        merged = re.sub(r'。+', '。', merged)  # 多个句号合并
        if not merged.endswith(("。", "！", "？")):
            merged += "。"

        return merged

    # ==================== 复合问题处理（主入口） ====================
    def _ask_multiple(self, query: str) -> str:
        """
        处理复合问题：
        1. 切分子问题
        2. 从第一个子问题提取景点名
        3. 为缺少景点名的子问题补全
        4. 逐个检索答案
        5. 合并成一段自然语言
        """
        sub_questions = self._split_questions(query)
        if len(sub_questions) <= 1:
            return self._single_ask(query)

        # ---- 1. 提取景点名（从第一个子问题中） ----
        scenic_name = self._extract_scenic_name(sub_questions[0])

        # ---- 2. 补全子问题并逐个检索 ----
        raw_answers = []
        for idx, sub_q in enumerate(sub_questions):
            # 如果不是第一个问题，且景点名存在，则补全
            if idx > 0 and scenic_name:
                enhanced_q = self._complete_sub_question(sub_q, scenic_name)
            else:
                enhanced_q = sub_q

            # 调用单问题检索
            ans = self._single_ask(enhanced_q)

            # 提取纯答案正文（去掉元数据和表情符号）
            clean = ans.split('\n')[0] if '\n' in ans else ans
            clean = re.sub(r'^[^\u4e00-\u9fa5a-zA-Z]+', '', clean).strip()
            raw_answers.append(clean)

        # ---- 3. 合并答案 ----
        merged_answer = self._merge_answers(sub_questions, raw_answers, scenic_name)

        return merged_answer

    # ==================== 对外接口 ====================
    def ask(self, query: str) -> str:
        """
        接受用户问题，自动识别并处理单问题或复合问题。
        """
        if not query.strip():
            return "请输入有效的问题。"

        sub_qs = self._split_questions(query)
        if len(sub_qs) > 1:
            return self._ask_multiple(query)
        else:
            return self._single_ask(query)


# ==================== 测试入口 ====================
if __name__ == "__main__":
    try:
        guide = AIGuideRAG(DB_PATH, enabled_only=True)
        print("\n" + "=" * 50)
        print("欢迎使用本地算法 AI 导游（输入 q 退出）")
        print("=" * 50)

        while True:
            query = input("\n👤 提问: ")
            if query.lower() == "q":
                break
            if not query.strip():
                continue
            answer = guide.ask(query)
            print(f"\n🤖 算法导游: \n{answer}")
            print("-" * 50)
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("发生错误，按回车键退出...")