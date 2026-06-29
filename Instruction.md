# 使用说明
本文档记录了项目在日常维护中的通用方法，方便调用。

### Python 依赖与虚拟环境配置

- 生成初始依赖清单（需提前全局安装 `pipreqs`）

  ```bash
  # . 代表文件路径
  pipreqs . --encoding=utf8 --force
  ```
- 创建 Python 隔离虚拟环境

  ```bash
  python -m venv .venv
  ```
- 激活虚拟环境

  > Windows 用户 (cmd)
  >

  ```bash
  .venv/Scripts/activate
  ```

  > Linux/Mac 用户
  >

  ```bash
  source .venv/bin/activate`
  ```
- 检查环境是否干净

  ```bash
  python -m pip freeze
  ```
- 升级 pip 到最新版本

  ```bash
  python -m pip install --upgrade pip
  ```
- 安装项目依赖

  ```bash
  python -m pip install -r requirements.txt
  ```

  > 清华镜像源：
  >
  > ```bash
  > python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  > ```
  >
- 导出当前环境所有包的精确版本（覆盖原 requirements.txt）

  ```bash
  python -m pip freeze > requirements.txt
  ```

> **提示**：若安装过程中遇到依赖冲突，可调整 `requirements.txt` 中的版本范围后重试安装。
> **建议**：因为 `pipreqs` 导出的依赖文件版本号不清晰，建议手动去除版本号再安装。