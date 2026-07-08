import sys

sys.dont_write_bytecode = True

from .db import LandmarkDB


class QASystem:
    def __init__(self):
        self.db = LandmarkDB()

    def keyword_match(self, question):
        """从数据库查询匹配的答案"""
        rows = self.db.get_all_knowledge()
        if not rows:
            return None

        question_lower = question.lower()

        for row in rows:
            # 关键词匹配
            keywords = row["keywords"]
            if keywords:
                kw_list = [kw.strip().lower() for kw in keywords.split(",")]
                for kw in kw_list:
                    if kw and kw in question_lower:
                        return row["answer"]

            # 完整问题包含匹配
            if row["question"] and row["question"] in question:
                return row["answer"]

        return None

    def get_answer(self, question):
        """主入口：使用本地数据库匹配"""
        local_answer = self.keyword_match(question)
        if local_answer:
            return {"answer": local_answer, "source": "local"}
        else:
            return {"answer": "抱歉，当前知识库中未找到相关问题的答案。", "source": "local"}