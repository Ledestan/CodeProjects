"""
项目名称: 星火云枢：班级智能协同平台
启动日期: 2026-06-02
"""

import sys

sys.dont_write_bytecode = True

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
