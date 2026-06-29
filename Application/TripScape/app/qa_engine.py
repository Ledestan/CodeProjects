import sys

sys.dont_write_bytecode = True

import requests

from .db import LandmarkDB


class QASystem:
    def __init__(self):
        self.api_key = "your_deepseek_api_key"  # API 密钥
        # 实例化数据库操作类
        self.db = LandmarkDB()

    def keyword_match(self, question):
        """从 MySQL 查询匹配的答案"""
        # 获取所有启用的知识条目
        rows = self.db.get_all_knowledge()
        if not rows:
            return None

        question_lower = question.lower()

        for row in rows:
            # 关键词匹配
            if row.get("keywords"):
                keywords = [kw.strip().lower() for kw in row["keywords"].split(",")]
                for kw in keywords:
                    if kw and kw in question_lower:
                        return row["answer"]

            # 完整问题包含匹配
            if row.get("question") and row["question"] in question:
                return row["answer"]

        return None

    def call_deepseek_api(self, question):
        """调用 DeepSeek API（兜底方案）"""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "你是一位世界著名地标和旅游景点的专家。"
            "请用准确、易懂的语言回答问题。"
            "如果问题涉及地标建筑或自然景观，请结合历史背景和文化意义详细说明。"
        )

        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "max_tokens": 500,
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"DeepSeek API 调用失败: {e}")
            return "抱歉，我现在无法回答这个问题，请稍后再试。"

    def get_answer(self, question):
        """主入口：先本地匹配，失败则调用 API"""
        # 尝试本地数据库匹配
        local_answer = self.keyword_match(question)
        if local_answer:
            return {"answer": local_answer, "source": "local"}

        # 兜底：调用 DeepSeek API
        api_answer = self.call_deepseek_api(question)
        return {"answer": api_answer, "source": "api"}
