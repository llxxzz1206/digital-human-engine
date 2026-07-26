"""直接测试 FastAPI 应用路由"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

print("FastAPI 应用路由:")
print(f"总路由数: {len(app.routes)}")

config_routes = []
for route in app.routes:
    if hasattr(route, 'path') and 'config' in route.path:
        config_routes.append(route.path)

print(f"\n包含 'config' 的路由数: {len(config_routes)}")
for route in config_routes:
    print(f"  - {route}")

print("\n所有 /admin 开头的路由:")
admin_routes = []
for route in app.routes:
    if hasattr(route, 'path') and route.path.startswith('/admin'):
        admin_routes.append(route.path)

for route in sorted(set(admin_routes)):
    print(f"  - {route}")