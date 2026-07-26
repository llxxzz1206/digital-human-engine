# P1 文档解析功能 - 测试报告

**日期：** 2026-07-19
**测试范围：** PDF/Word/TXT/MD 文档解析 + 自适应分块 + Embedding 失败处理
**对应验收项：** M-02（知识库管理）

---

## 测试环境

- OS: Windows 10
- Python: 3.11+ (uv 管理)
- 新增依赖: pymupdf 1.28.0, python-docx 1.2.0, python-multipart 0.0.32, lxml 6.1.1

## 测试结果

| # | 测试项 | 方法 | 结果 |
|---|--------|------|------|
| 1 | TXT 纯文本解析 | 创建临时 .txt 文件，调用 parse_document | PASS |
| 2 | Markdown 解析 | 创建临时 .md 文件，验证标题和正文提取 | PASS |
| 3 | PDF 解析 | PyMuPDF 生成含文本 PDF，验证提取结果和页数 | PASS |
| 4 | DOCX 解析（含表格） | python-docx 生成含段落+表格的文档，验证全部提取 | PASS |
| 5 | 不支持格式拒绝 | 上传 .xlsx 文件，验证抛出 DocumentParseError | PASS |
| 6 | 文件不存在处理 | 传入不存在路径，验证错误提示 | PASS |
| 7 | 自适应分块大小 | 验证 PDF=800, DOCX=600, TXT=500, MD=600 | PASS |
| 8 | 段落智能分块 | 1058 字符多段落文本 → 3 块，按段落边界切分 | PASS |

**通过率：8/8 (100%)**

## 新增 API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| /api/knowledge/upload | POST | 上传文件（multipart/form-data）→ 解析 → 分块 → 向量化 → Milvus |
| /api/knowledge/build-text | POST | 纯文本直接构建知识库 |
| /api/knowledge/supported-formats | GET | 查询支持的文件格式 |

## 代码变更

| 文件 | 变更 |
|------|------|
| app/rag/document_parser.py | 新增：文档解析器（PDF/Word/TXT/MD） |
| app/rag/knowledge_builder.py | 重写：新增 build_from_file()、段落智能分块、Embedding 失败过滤 |
| app/rag/embedding.py | 修复：失败返回 None 而非随机向量（P4） |
| app/api/routes/knowledge.py | 新增：知识库管理 API 路由 |
| app/main.py | 注册 knowledge 路由 |
| pyproject.toml | 新增 pymupdf, python-docx, python-multipart 依赖 |
| test_document_parser.py | 新增：8 项单元测试 |

## 遗留问题

- .doc（旧版 Word）不支持，需用户转为 .docx（已在错误提示中说明）
- 扫描件 PDF（纯图片）无法提取文本，会返回明确错误
- 端到端测试（上传 → Milvus 写入 → 检索验证）需要 Milvus 和 Embedding API 在线，留待集成测试
