"""验证 admin.py 路由是否正确加载"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.api.routes import admin

print("admin.router 类型:", type(admin.router))
print("admin.router.prefix:", admin.router.prefix)

# 获取所有路由
routes = []
for route in admin.router.routes:
    if hasattr(route, 'path'):
        routes.append(route.path)

print(f"\n路由数量: {len(routes)}")
print("\n包含 config 的路由:")
for route in routes:
    if 'config' in route:
        print(f"  - {route}")

print("\n前10个路由:")
for route in routes[:10]:
    print(f"  - {route}")