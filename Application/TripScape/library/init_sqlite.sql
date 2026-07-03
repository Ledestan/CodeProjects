-- 删除旧表（如果存在）
DROP TABLE IF EXISTS heritage_items;
DROP TABLE IF EXISTS knowledge_base;

-- 验证插入数据
SELECT * FROM heritage_items;
SELECT * FROM knowledge_base;

-- 创建地标信息表
CREATE TABLE heritage_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 主键，自增
    target_id VARCHAR(50) UNIQUE NOT NULL, -- 地标唯一标识
    name VARCHAR(100) NOT NULL,            -- 地标中文名称
    year VARCHAR(50),                      -- 建造年代或时期
    description TEXT,                      -- 详细描述
    location VARCHAR(200),                 -- 地理位置
    current_status VARCHAR(200),           -- 当前保护状态或用途
    image_path VARCHAR(255),               -- 模板图片文件名
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- 记录创建时间
);

-- 创建问答知识库表
CREATE TABLE knowledge_base (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 主键，自增
    question VARCHAR(255) NOT NULL,        -- 典型问题
    keywords TEXT,                         -- 逗号分隔的关键词
    answer TEXT NOT NULL,                  -- 答案内容
    enabled INTEGER DEFAULT 1,             -- 是否启用（1启用，0禁用）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- 创建时间
);
