# P2-P6 Python 功能完善 - 测试报告

**日期：** 2026-07-19
**测试范围：** MCP工具调用、多轮上下文、并发修复、TTS缓存
**对应验收项：** F-03（智能对话）、S-01（并发支撑）、S-03（TTS并发控制）

---

## 测试环境

- OS: Windows 10
- Python: 3.11+ (uv 管理)
- 验证方式：模块导入测试 + 代码逻辑审查（端到端测试需 LLM API + Redis + PostgreSQL 在线）

## 变更清单

| # | 任务 | 文件 | 变更内容 |
|---|------|------|----------|
| P2 | MCP tool_executor 接入 | app/workflow/nodes.py | reply_generator 使用 stream_chat_with_tools() agentic loop；新增 _mcp_tool_executor 桥接 MCP registry；新增 get_all_available_tools() 合并内置工具 |
| P3 | 多轮上下文窗口扩展 | app/workflow/nodes.py, app/config/settings.py | 窗口从 6 条(3轮)扩展到 20 条(10轮)，可配置 RAG__CONTEXT_WINDOW；超出部分 LLM 摘要压缩 |
| P5 | WS 并发串流修复 | app/workflow/nodes.py | _current_send_func 从全局变量改为 contextvars.ContextVar，每个 asyncio Task 独立 |
| P6 | TTS 缓存补全 | app/workflow/nodes.py | _stream_tts_sentence 先查缓存(命中一次性发送)，未命中流式合成后异步写入缓存 |

## 测试结果

| # | 测试项 | 方法 | 结果 |
|---|--------|------|------|
| 1 | 模块导入完整性 | `from app.main import app` + 所有新增函数导入 | PASS |
| 2 | get_all_available_tools 合并逻辑 | 验证内置 MCP 工具(echo/avatar_action/current_time)被正确转换为 OpenAI function calling 格式 | PASS |
| 3 | ContextVar 隔离性 | 代码审查：每个 asyncio.create_task 自动继承独立 context，不同 session 的 send_func 互不干扰 | PASS (逻辑验证) |
| 4 | TTS 缓存流程 | 代码审查：get→命中直接发送 / 未命中→流式合成→异步 put | PASS (逻辑验证) |
| 5 | 摘要压缩降级 | 代码审查：LLM 调用失败时降级为拼接最后 4 条消息文本 | PASS (逻辑验证) |
| 6 | _send 异常安全 | 代码审查：send 失败只 log 不抛异常，不中断 workflow | PASS |

**通过率：6/6 (100%)**

## 端到端测试前置条件

以下测试需要完整环境（LLM API + Redis + PostgreSQL + Milvus），留待集成测试阶段：

- 对话中问"现在几点" → 触发 current_time 工具 → LLM 整合结果回复
- 连续对话 10+ 轮后引用第 1 轮内容 → 摘要压缩生效
- 两个浏览器 tab 同时对话 → 各自收到正确回复不串流
- 同一句话第二次 TTS 延迟明显降低 → 缓存命中

## 配置说明

新增环境变量（可选）：
```
RAG__CONTEXT_WINDOW=20  # LLM 上下文窗口大小（消息条数），默认 20 = 10 轮对话
```
