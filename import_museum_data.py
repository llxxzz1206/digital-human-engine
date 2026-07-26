"""导入博物馆场景知识库（知识文档 + FAQ）
用法: uv run python import_museum_data.py
前提: Python AI Engine 运行中 (localhost:8000), Milvus 运行中
"""
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

API = "http://localhost:8000"
SKILL_ID = "museum"


def api_post(path: str, data: dict) -> dict:
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ========== 博物馆知识文档（导入 skill_museum 集合） ==========
MUSEUM_DOCUMENTS = [
    {
        "text": """城市历史博物馆 — 一楼大厅与临时展厅
一楼大厅：服务台（门票验证、语音导览租借、存包柜）、文创商店、休息咖啡区、无障碍通道。
临时展厅A（东侧300㎡）：每季度更换主题特展，近期展出"丝路遗珍——丝绸之路文物精品展"。
临时展厅B（西侧200㎡）：当代艺术邀请展、市民摄影作品展。
多功能报告厅：周末公益讲座、亲子手工坊、学术研讨会（可容纳120人）。""",
        "metadata": {"museum": "城市历史博物馆", "floor": "1楼", "category": "大厅与临展"},
    },
    {
        "text": """城市历史博物馆 — 二楼 古代文明展厅
第一展厅"远古足迹"：旧石器时代打制石器、新石器时代彩陶、骨器、玉琮，展示本地先民从采集狩猎到农耕定居的演变。重点藏品：距今8000年的刻画纹陶片（镇馆之宝之一）。
第二展厅"青铜时代"：商周青铜礼器、兵器、车马器。重点藏品：西周饕餮纹方鼎（国家一级文物）、战国错金银铜壶。
第三展厅"汉唐风华"：汉代画像砖、陶俑、铜镜，唐代三彩器、金银器、丝绸残片。重点藏品：唐三彩骆驼载乐俑。""",
        "metadata": {"museum": "城市历史博物馆", "floor": "2楼", "category": "古代文明"},
    },
    {
        "text": """城市历史博物馆 — 三楼 近现代历史展厅
第四展厅"百年风云"（1840-1949）：鸦片战争至新中国成立，展出近代条约文书、革命文物、抗战实物、老照片。重点藏品：1911年起义军政府告示原件。
第五展厅"城市记忆"（1949至今）：老城墙砖、旧城门照片、改革开放初期个体户营业执照、80年代家电实物。互动区：老式电话亭、黑白电视机体验。
第六展厅"非遗活态"：本地非物质文化遗产展示——剪纸、泥塑、皮影戏道具，定期有传承人现场演示。""",
        "metadata": {"museum": "城市历史博物馆", "floor": "3楼", "category": "近现代历史"},
    },
    {
        "text": """城市历史博物馆 — 四楼 自然与科技展厅
第七展厅"自然奥秘"：本地地质标本、古生物化石（三叶虫、菊石、恐龙骨骼模型）、矿物晶体、动植物标本。互动装置：地震模拟平台、VR深海探险。
第八展厅"科技之光"：本地工业发展史，展出第一台国产机床模型、航天零件实物、新能源技术展板。儿童科学角：简单机械实验、电路拼装。
天文球幕影院（四楼西侧）：直径12米球幕，每日放映3场（10:30/14:00/15:30），每场30分钟，免费但需预约。""",
        "metadata": {"museum": "城市历史博物馆", "floor": "4楼", "category": "自然与科技"},
    },
    {
        "text": """城市历史博物馆 — 参观路线推荐
经典路线（2小时）：1楼大厅→2楼古代文明（重点看青铜方鼎和三彩骆驼）→3楼近现代历史→4楼自然奥秘→1楼文创商店。
亲子路线（1.5小时）：1楼大厅→4楼自然奥秘+儿童科学角→天文球幕影院→2楼远古足迹（看彩陶）→1楼多功能厅（周末手工坊）。
深度路线（3.5小时）：按楼层顺序逐厅参观，建议租借语音导览（20元/台），每层休息区有饮水机。
无障碍路线：全馆电梯通达，轮椅可从1楼服务台免费借用，各展厅入口有无障碍坡道。""",
        "metadata": {"museum": "城市历史博物馆", "floor": "全馆", "category": "参观路线"},
    },
    {
        "text": """城市历史博物馆 — 镇馆之宝（重点藏品讲解）
1. 刻画纹陶片（新石器时代，距今约8000年）：本地出土最早的陶器残片，表面刻有几何纹饰，是研究本地先民制陶工艺的关键实物。
2. 西周饕餮纹方鼎（国家一级文物）：通高42厘米，重12.5公斤，四面饰饕餮纹，内壁铸铭文16字，记载了一次祭祀活动。
3. 唐三彩骆驼载乐俑：高58厘米，骆驼背上有5个乐俑演奏不同乐器，是丝绸之路文化交流的生动见证。
4. 1911年起义军政府告示原件：辛亥革命时期本地军政府发布的第一份安民告示，纸质泛黄但字迹清晰。
5. 战国错金银铜壶：通体错金银云纹，工艺精湛，是战国时期金属镶嵌工艺的代表作。""",
        "metadata": {"museum": "城市历史博物馆", "floor": "全馆", "category": "重点藏品"},
    },
]


# ========== 博物馆 FAQ（导入 faq_museum 集合） ==========
MUSEUM_FAQS = [
    # --- 基本信息 ---
    {"question": "博物馆几点开门", "answer": "开馆时间为周二至周日9:00-17:00（16:30停止入馆），周一闭馆（法定节假日除外）。"},
    {"question": "周一开门吗", "answer": "周一闭馆维护（法定节假日除外）。如遇节假日调休，请关注公众号公告。"},
    {"question": "门票多少钱", "answer": "常设展览免费开放，凭身份证或预约码入馆。临时特展可能收费（一般20-50元），天文球幕影院免费但需预约。"},
    {"question": "需要预约吗", "answer": "建议提前在博物馆微信公众号预约，凭预约码入馆。未预约也可现场刷身份证入馆，但节假日人多时可能限流。"},
    {"question": "怎么预约", "answer": "在博物馆微信公众号点击'参观预约'，选择日期和时段，填写姓名和身份证号即可。每天限约3000人。"},
    {"question": "可以带小孩来吗", "answer": "当然可以。4楼有儿童科学角和互动装置，周末1楼多功能厅有亲子手工坊。1.2米以下儿童无需预约，由家长陪同入馆。"},
    {"question": "可以带宠物吗", "answer": "不可以。除导盲犬外，宠物不能入馆。馆外有临时宠物寄存笼（免费，限时2小时）。"},

    # --- 交通与停车 ---
    {"question": "怎么坐地铁到博物馆", "answer": "地铁3号线'博物馆站'A出口，步行约300米即到。也可乘1号线'文化广场站'换乘3号线。"},
    {"question": "有停车场吗", "answer": "有。博物馆地下停车场（B1层），入口在馆区东侧。前2小时免费，之后每小时4元。周末车位紧张建议公共交通。"},
    {"question": "有自行车停放处吗", "answer": "有。馆区南门和北门各有一处非机动车停放区，免费停放。"},

    # --- 馆内服务 ---
    {"question": "有语音导览吗", "answer": "有。1楼服务台租借语音导览器，20元/台（押金100元），支持中英文。也可在公众号使用手机扫码听讲解（免费）。"},
    {"question": "有讲解员吗", "answer": "有。每天10:00和14:30各有一场免费定时讲解（1楼大厅集合）。团体预约讲解需提前3天在公众号申请。"},
    {"question": "可以拍照吗", "answer": "常设展厅允许拍照（禁止闪光灯和三脚架）。临时特展部分展品禁止拍照，以现场标识为准。"},
    {"question": "有存包的地方吗", "answer": "有。1楼大厅东侧有免费电子存包柜，大件行李（超过40cm）必须寄存。"},
    {"question": "有餐厅吗", "answer": "1楼有咖啡轻食区，提供简餐、饮品和甜点。馆外50米有多家餐厅。"},
    {"question": "有母婴室吗", "answer": "有。1楼服务台旁和4楼电梯旁各有一间母婴室，配有哺乳椅和尿布台。"},
    {"question": "有无障碍设施吗", "answer": "全馆无障碍。电梯通达每层，各展厅入口有坡道。1楼服务台可免费借用轮椅。"},
    {"question": "可以带食物进馆吗", "answer": "展厅内禁止饮食。1楼咖啡区和各层休息区可以喝水、吃简餐。"},

    # --- 展览与活动 ---
    {"question": "现在有什么特展", "answer": "当前临时展厅A展出'丝路遗珍——丝绸之路文物精品展'（至本月底），临时展厅B为当代艺术邀请展。详情见公众号'当前展览'。"},
    {"question": "有讲座活动吗", "answer": "有。每周六下午2点在1楼多功能报告厅有公益讲座（免费，无需预约，先到先得）。亲子手工坊每周日上午10点，需公众号预约。"},
    {"question": "球幕影院怎么预约", "answer": "在公众号'参观预约'中选择'天文球幕'，选择场次（10:30/14:00/15:30）。每场限40人，建议提前1天预约。"},
    {"question": "团体参观怎么预约", "answer": "20人以上团体需提前3天在公众号'团体预约'提交申请，或致电0571-8888-6666。团体可享免费讲解服务。"},
    {"question": "有志愿者讲解吗", "answer": "有。每天定时讲解（10:00/14:30）由志愿者担任。也可在1楼服务台咨询是否有志愿者在岗。"},

    # --- 重点藏品 ---
    {"question": "镇馆之宝是什么", "answer": "五大镇馆之宝：8000年刻画纹陶片、西周饕餮纹方鼎（国家一级文物）、唐三彩骆驼载乐俑、1911年起义军政府告示、战国错金银铜壶。"},
    {"question": "青铜方鼎在哪个展厅", "answer": "西周饕餮纹方鼎在2楼第二展厅'青铜时代'，进门右手边独立展柜，有专题灯光照明。"},
    {"question": "三彩骆驼在哪看", "answer": "唐三彩骆驼载乐俑在2楼第三展厅'汉唐风华'，展厅中央独立展台。"},
    {"question": "恐龙化石在哪", "answer": "恐龙骨骼模型在4楼第七展厅'自然奥秘'，展厅入口处即可看到大型恐龙骨架。"},
    {"question": "非遗展示在哪", "answer": "非物质文化遗产展示在3楼第六展厅'非遗活态'，有剪纸、泥塑、皮影戏道具，周末有传承人现场演示。"},

    # --- 其他 ---
    {"question": "可以带行李箱吗", "answer": "大件行李（超过40cm）需在1楼存包柜寄存，不能带入展厅。小型背包可以背入。"},
    {"question": "有WiFi吗", "answer": "有。连接'Museum-Free'即可免费使用，无需密码。"},
    {"question": "可以带饮料进展厅吗", "answer": "密封瓶装水可以带入展厅，但开盖饮料和食物不行。各层有休息区可以喝水。"},
    {"question": "博物馆商店卖什么", "answer": "1楼文创商店有馆藏文物复刻品、文创文具、丝巾、冰箱贴、明信片等。热门商品：青铜方鼎冰箱贴、三彩骆驼摆件。"},
    {"question": "可以当志愿者吗", "answer": "可以。每年3月和9月招募志愿者，在公众号'志愿者招募'报名。要求每月至少服务2次，提供培训和工作餐。"},
    {"question": " lost and found在哪", "answer": "失物招领在1楼服务台。拾到物品请交至服务台登记，丢失物品可致电0571-8888-6666查询。"},
]


async def import_knowledge():
    """导入知识文档到 skill_museum 集合"""
    from app.rag.knowledge_builder import knowledge_builder

    print(f"📥 导入博物馆知识文档 ({len(MUSEUM_DOCUMENTS)} 篇) → skill_{SKILL_ID}")
    count = await knowledge_builder.build(SKILL_ID, MUSEUM_DOCUMENTS)
    print(f"✅ 知识文档导入完成: {count} 个向量切片")


def import_faqs():
    """导入 FAQ 到 faq_museum 集合"""
    print(f"\n📥 导入博物馆 FAQ ({len(MUSEUM_FAQS)} 条) → faq_{SKILL_ID}")

    try:
        result = api_post("/api/faq/import", {
            "skill_id": SKILL_ID,
            "items": MUSEUM_FAQS,
        })
        imported = result.get("imported", 0)
        print(f"✅ FAQ 批量导入完成: {imported}/{len(MUSEUM_FAQS)} 条")
    except Exception as e:
        print(f"⚠️ 批量接口失败({e})，逐条导入...")
        promoted = 0
        for i, faq in enumerate(MUSEUM_FAQS):
            try:
                result = api_post("/api/faq/promote", {
                    "skill_id": SKILL_ID,
                    "question": faq["question"],
                    "answer": faq["answer"],
                })
                if result.get("status") == "promoted":
                    promoted += 1
                if (i + 1) % 10 == 0:
                    print(f"  进度: {i+1}/{len(MUSEUM_FAQS)}")
                time.sleep(0.2)
            except Exception as ex:
                print(f"  [{i+1}] ERROR: {ex}")
        print(f"✅ 逐条导入完成: {promoted}/{len(MUSEUM_FAQS)} 成功")


async def verify():
    """验证检索效果"""
    from app.rag.retriever import rag_retriever
    from app.rag.reranker import reranker

    test_queries = [
        "博物馆几点关门",
        "青铜方鼎在哪里看",
        "有停车场吗",
        "恐龙化石在哪个展厅",
        "怎么预约讲解",
    ]

    print("\n🔍 验证检索效果：")
    for query in test_queries:
        results = await rag_retriever.search(query, skill_ids=[SKILL_ID], top_k=3)
        if results:
            top = results[0]
            score = top.get("score", 0)
            text_preview = top.get("text", "")[:50]
            print(f"  Q: {query}")
            print(f"  → score={score:.3f}: {text_preview}...")
        else:
            print(f"  Q: {query} → 无结果")
    print()


async def main():
    await import_knowledge()
    import_faqs()
    await verify()
    print("🎉 博物馆知识库导入全部完成！")


if __name__ == "__main__":
    asyncio.run(main())
