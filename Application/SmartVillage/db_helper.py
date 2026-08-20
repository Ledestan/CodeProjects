import sqlite3
from datetime import datetime

import pandas as pd

DB_FILE = "workorders.db"


def get_conn():
    """获取数据库连接，返回连接对象"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库：仅建表，不插入任何预设数据"""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_no TEXT NOT NULL UNIQUE,
            reporter TEXT NOT NULL,
            phone TEXT,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT '待处理',
            handler TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_all_tickets(status=None, category=None):
    """查询工单，支持按状态和类别筛选"""
    conn = get_conn()
    query = "SELECT * FROM tickets WHERE 1=1"
    params = []
    if status and status != "全部":
        query += " AND status = ?"
        params.append(status)
    if category and category != "全部":
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY created_at DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_ticket_by_no(ticket_no):
    """根据编号查询单条工单"""
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM tickets WHERE ticket_no = ?", conn, params=[ticket_no]
    )
    conn.close()
    return df.iloc[0] if not df.empty else None


def add_ticket(reporter, phone, category, description):
    """新增工单，返回生成的工单编号"""
    conn = get_conn()
    cursor = conn.cursor()
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    prefix = f"T{date_str}"
    cursor.execute(
        "SELECT MAX(CAST(SUBSTR(ticket_no, 9) AS INTEGER)) FROM tickets WHERE ticket_no LIKE ?",
        (f"{prefix}%",),
    )
    max_seq = cursor.fetchone()[0] or 0
    new_seq = str(max_seq + 1).zfill(3)
    ticket_no = f"{prefix}{new_seq}"
    time_str = now.strftime("%Y-%m-%d %H:%M")
    cursor.execute(
        """INSERT INTO tickets 
           (ticket_no, reporter, phone, category, description, status, handler, created_at, updated_at) 
           VALUES (?, ?, ?, ?, ?, '待处理', '', ?, ?)""",
        (ticket_no, reporter, phone, category, description, time_str, time_str),
    )
    conn.commit()
    conn.close()
    return ticket_no


def update_status(ticket_no, new_status, handler=""):
    """更新工单状态（派单/办结）"""
    conn = get_conn()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute(
        "UPDATE tickets SET status = ?, handler = ?, updated_at = ? WHERE ticket_no = ?",
        (new_status, handler, now, ticket_no),
    )
    conn.commit()
    conn.close()


def delete_ticket(ticket_no):
    """删除工单（仅管理员）"""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tickets WHERE ticket_no = ?", (ticket_no,))
    conn.commit()
    conn.close()


def count_by_status():
    """统计各状态数量"""
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT status, COUNT(*) as count FROM tickets GROUP BY status", conn
    )
    conn.close()
    result = {"待处理": 0, "处理中": 0, "已办结": 0}
    for _, row in df.iterrows():
        result[row["status"]] = row["count"]
    return result


def count_by_category():
    """统计各类别未办结数量"""
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT category, COUNT(*) as count FROM tickets WHERE status != '已办结' GROUP BY category",
        conn,
    )
    conn.close()
    return df


def get_recent_tickets(limit=5):
    """获取最新N条工单（用于大屏）"""
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT ticket_no, reporter, category, description, status, created_at FROM tickets ORDER BY created_at DESC LIMIT ?",
        conn,
        params=[limit],
    )
    conn.close()
    return df
