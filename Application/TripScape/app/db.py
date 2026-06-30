import sqlite3


class LandmarkDB:
    def __init__(self, path="data/landmark.db"):
        self.path = path
        self.connection = None
        self.cursor = None

    def connect(self):
        """建立数据库连接，设置返回结果为字典格式，若已存在连接则直接返回"""
        if self.connection:
            return
        try:
            self.connection = sqlite3.connect(self.path)
            # 设置 row_factory 使返回的行支持字典访问（列名索引）
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {e}")

    def get_heritage_info(self, target_id=None):
        """获取地标信息"""
        self.connect()
        try:
            if target_id:
                sql = "SELECT * FROM heritage_items WHERE target_id = ?"
                self.cursor.execute(sql, (target_id,))
                return self.cursor.fetchone()
            else:
                sql = "SELECT * FROM heritage_items"
                self.cursor.execute(sql)
                return self.cursor.fetchall()
        except Exception as e:
            print(f"查询地标信息失败: {e}")
            return None

    def get_all_heritage_targets(self):
        """获取所有地标的 target_id 和图片路径（用于加载模板图片）"""
        self.connect()
        try:
            sql = "SELECT target_id, image_path FROM heritage_items WHERE image_path IS NOT NULL"
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"查询地标模板失败: {e}")
            return []

    def get_all_knowledge(self):
        """获取所有启用的知识库条目（用于关键词匹配问答）"""
        self.connect()
        try:
            sql = "SELECT id, question, keywords, answer FROM knowledge_base WHERE enabled = 1"
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"查询知识库失败: {e}")
            return []

    def get_answer_by_question(self, question):
        """根据完整问题精确匹配答案"""
        self.connect()
        try:
            sql = "SELECT answer FROM knowledge_base WHERE question = ? AND enabled = 1"
            self.cursor.execute(sql, (question,))
            row = self.cursor.fetchone()
            return row["answer"] if row else None
        except Exception as e:
            print(f"精确查询失败: {e}")
            return None

    def close(self):
        """关闭数据库连接和游标"""
        if self.connection:
            self.cursor.close()
            self.connection.close()
            print("数据库连接已关闭")
