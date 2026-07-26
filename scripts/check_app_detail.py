"""检查 FastAPI 应用完整路由"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

print("FastAPI 应用信息:")
print(f"应用类型: {type(app)}")
print(f"应用标题: {app.title}")
print(f"总路由数: {len(app.routes)}")

print("\n所有路由:")
for i, route in enumerate(app.routes, 1):
    if hasattr(route, 'path'):
        methods = getattr(route, 'methods', ['GET'])
        print(f"{i}. {route.path} - {methods}")

print("\n检查 router 属性:")
if hasattr(app, 'router'):
    print(f"app.router 类型: {type(app.router)}")
else:
    print("app 没有 router 属性")

print("\n检查 routes 属性:")
print(f"app.routes 类型: {type(app.routes)}")