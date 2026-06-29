import pymysql
from pymysql.err import OperationalError


class LandmarkDB:
    def __init__(self):
        self.host = "localhost"  # MySQL 服务器地址
        self.user = "root"  # 用户名
        self.password = "Root@123456"  # 密码
        self.db = "landmark_db"  # 数据库名称
        self.charset = "utf8mb4"  # 编码格式

        self.connection = None  # 数据库连接
        self.cursor = None  # 游标

    def connect(self):
        """建立数据库连接"""
        if self.connection and self.connection.open:
            return

        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.db,
                charset=self.charset,
                cursorclass=pymysql.cursors.DictCursor,  # 返回字典格式
            )
            self.cursor = self.connection.cursor()
            print("数据库连接成功")
        except OperationalError as e:
            print(f"数据库连接失败: {e}")
        except Exception as e:
            print(f"连接错误: {e}")

    # ---------- 地标信息相关 ----------
    def get_heritage_info(self, target_id=None):
        """
        获取地标信息
        - 若指定 target_id, 返回单个地标信息（字典）
        - 若未指定，返回所有地标信息（列表）
        """
        self.connect()
        try:
            if target_id:
                sql = "SELECT * FROM heritage_items WHERE target_id = %s"
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
        """获取所有地标的 target_id 和图片路径（用于加载模板）"""
        self.connect()
        try:
            sql = "SELECT target_id, image_path FROM heritage_items WHERE image_path IS NOT NULL"
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"查询地标模板失败: {e}")
            return []

    # ---------- 问答知识库相关 ----------
    def get_all_knowledge(self):
        """获取所有启用的知识库条目（用于关键词匹配）"""
        self.connect()
        try:
            sql = "SELECT id, question, keywords, answer FROM knowledge_base WHERE enabled = 1"
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"查询知识库失败: {e}")
            return []

    def get_answer_by_question(self, question):
        """根据完整问题精确匹配（备用）"""
        self.connect()
        try:
            sql = (
                "SELECT answer FROM knowledge_base WHERE question = %s AND enabled = 1"
            )
            self.cursor.execute(sql, (question,))
            row = self.cursor.fetchone()
            return row["answer"] if row else None
        except Exception as e:
            print(f"精确查询失败: {e}")
            return None

    # ---------- 聊天记录 ----------
    def save_chat_history(
        self, session_id, user_input, bot_response, input_type="text"
    ):
        """保存对话记录"""
        self.connect()
        try:
            sql = """
                INSERT INTO chat_history (session_id, user_input, bot_response, input_type)
                VALUES (%s, %s, %s, %s)
            """
            self.cursor.execute(sql, (session_id, user_input, bot_response, input_type))
            self.connection.commit()
            return True
        except Exception as e:
            print(f"保存聊天记录失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.connection and self.connection.open:
            self.cursor.close()
            self.connection.close()
            print("数据库连接已关闭")
