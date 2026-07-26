"""管理后台路由前缀约定测试

踩过的坑：faq.py / chat_logs.py 曾自带 /api 前缀，而 vite 与 nginx
代理都会剥掉 /api 再转发（约定"后端路由不带 /api"），导致管理后台
的对话日志、FAQ 管理两个 Tab 在浏览器里全部 404 → No data。

本测试通过 OpenAPI schema 检查全量路径（懒加载的 _IncludedRouter 也会
被展开；不需要启动 lifespan、不依赖任何服务），谁再把 /api 前缀加回去
就会红。
"""

from app.main import app

PATHS = set(app.openapi()["paths"].keys())


def test_no_route_carries_api_prefix():
    api_routes = [p for p in PATHS if p.startswith("/api")]
    assert not api_routes, f"代理会剥掉 /api，这些路由在浏览器里会 404: {api_routes}"


def test_admin_tab_routes_exist_at_proxy_stripped_paths():
    # 对话日志 Tab
    assert "/chat-logs/dates" in PATHS
    assert "/chat-logs" in PATHS
    # FAQ 管理 Tab
    assert "/faq/candidates" in PATHS
    assert "/faq/list" in PATHS
    assert "/faq/approve" in PATHS
    assert "/faq/reject" in PATHS
    # skill 管理 Tab
    assert "/admin/skill/list" in PATHS
