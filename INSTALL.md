# 项目环境初始化流程

本项目使用 Python 和 pip 管理依赖。以下步骤按顺序执行即可搭建开发环境。

1. 生成初始依赖清单（需提前全局安装 `pipreqs`）
   ```bash
   pipreqs . --encoding=utf8 --force
   ```
2. 创建 Python 隔离虚拟环境
   ```bash
   python -m venv .venv
   ```
3. 激活虚拟环境
   > Windows 用户 (cmd)
   ```bash
   .venv\Scripts\activate
   ```

   > Linux/Mac 用户
   ```bash
   source .venv/bin/activate`
   ```
4. 检查环境是否干净
   ```bash
   python -m pip freeze
   ```
5. 升级 pip 到最新版本
   ```bash
   python -m pip install --upgrade pip
   ```
6. 安装项目依赖
   ```bash
   python -m pip install -r requirements.txt
   ```
   > 清华镜像源：
   > ```bash
   > python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   > ```
7. 导出当前环境所有包的精确版本（覆盖原 requirements.txt）
   ```bash
   python -m pip freeze > requirements.txt
   ```

> **提示**：若安装过程中遇到依赖冲突，可调整 `requirements.txt` 中的版本范围后重试安装。
