# 使用说明

本文档记录了项目在日常维护中的通用方法，方便调用。

## Python 依赖与虚拟环境配置

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

---

## 数据库命令模板

### MySQL

- 切换到目标数据库（`*` 代表数据库名称）
  ```SQL
  USE *;
  ```

- 查看当前数据库中的所有表
  ```SQL
  SHOW TABLES;
  ```

- 查看表结构（`*` 代表表名）
  ```SQL
  DESC *;
  ```

- 清空表数据并重置自增 ID（`*` 代表表名）
  ```SQL
  TRUNCATE TABLE *;
  ```

- 删除整张表（`*` 代表表名）
  ```SQL
  DROP TABLE IF EXISTS *;
  ```

- 查询表中所有数据（`*` 代表表名）
  ```SQL
  SELECT * FROM *;
  ```

- 查询前 n 条数据（`*` 代表表名，`n` 为数字）
  ```SQL
  SELECT * FROM * LIMIT n;
  ```

- 插入数据（`*` 代表表名，字段和值自行替换）
  ```SQL
  INSERT INTO * (字段1, 字段2) VALUES (值1, 值2);
  ```

- 更新数据（`*` 代表表名，条件和赋值自行替换）
  ```SQL
  UPDATE * SET 字段 = 新值 WHERE 条件;
  ```

- 删除数据（`*` 代表表名，条件自行替换）
  ```SQL
  DELETE FROM * WHERE 条件;
  ```

- 查看当前连接的数据库名称
  ```SQL
  SELECT DATABASE();
  ```

- 建表
  ```SQL
  CREATE TABLE IF NOT EXISTS 表名 (
      id INT AUTO_INCREMENT PRIMARY KEY,
      字段1 VARCHAR(255) NOT NULL,
      字段2 INT DEFAULT 0,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  ) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;
  ```

---

### SQLite

- 查看当前连接的所有数据库文件（固定命令，无需替换）
  ```SQL
  PRAGMA database_list;
  ```
  > 如果需要查看单个数据库文件路径，可执行 `PRAGMA *;`（`*` 替换为数据库别名，如 `main`），但通常直接用 `database_list` 更直观。

- 查看当前数据库中的所有表
  ```SQL
  SELECT name FROM sqlite_master WHERE type = 'table';
  ```

- 查看表结构（`*` 代表表名）
  ```SQL
  PRAGMA table_info(*);
  ```

- 清空表数据（`*` 代表表名）
  ```SQL
  TRUNCATE TABLE *;
  ```
  或（若 `TRUNCATE` 不支持，使用此备用方案）
  ```SQL
  DELETE FROM *;
  ```

- 删除整张表（`*` 代表表名）
  ```SQL
  DROP TABLE IF EXISTS *;
  ```

- 查询表中所有数据（`*` 代表表名）
  ```SQL
  SELECT * FROM *;
  ```

- 查询前 n 条数据（`*` 代表表名，`n` 为数字）
  ```SQL
  SELECT * FROM * LIMIT n;
  ```

- 插入数据（`*` 代表表名，字段和值自行替换）
  ```SQL
  INSERT INTO * (字段1, 字段2) VALUES (值1, 值2);
  ```

- 更新数据（`*` 代表表名，条件和赋值自行替换）
  ```SQL
  UPDATE * SET 字段 = 新值 WHERE 条件;
  ```

- 删除数据（`*` 代表表名，条件自行替换）
  ```SQL
  DELETE FROM * WHERE 条件;
  ```

- 建表
  ```SQL
  CREATE TABLE IF NOT EXISTS 表名 (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      字段1 TEXT NOT NULL,
      字段2 INTEGER DEFAULT 0,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
  );
  ```