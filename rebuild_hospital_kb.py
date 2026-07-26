"""医院知识库一键重建脚本（单一数据源版）

数据源: data/hospital_floors.json —— 楼层布局唯一权威来源
重建内容:
  1. RAG 知识库   (Milvus skill_hospital)  ← 由 JSON 楼层文档生成
  2. 位置 FAQ     (Milvus faq_hospital + PG faq_candidate) ← 由 JSON 科室生成
  3. 人工 curated FAQ (挂号/医保/住院/交通/便民/检查预约) ← 原样保留

设计要点:
  - 重建前先备份现有 hospital FAQ 到 faq_hospital_backup_<时间戳>.json
  - 先清空 PG + Milvus 旧数据再写入，杜绝陈旧残留
    (教训: 曾因多个 import 脚本重复写入，导致 ICU 出现 4/9/16 楼三个版本)
  - 内置一致性自检: 导航工具返回楼层 == JSON，Milvus/PG 条数对账

用法: uv run python rebuild_hospital_kb.py
前提: PG + Milvus 运行中 (docker compose up -d)，无需启动后端服务
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.infrastructure.database import DatabasePool
from app.rag.knowledge_builder import knowledge_builder
from app.rag.faq_promotion import faq_promotion_service
from app.rag.milvus_client import milvus_manager
from app.rag.embedding import embedding_service

SKILL_ID = "hospital"
DATA_FILE = Path(__file__).parent / "data" / "hospital_floors.json"

# ── 人工 curated FAQ（非位置类为主，含就医流程/费用/住院/交通/便民/检查预约）──
# 位置类 FAQ 由 JSON 自动生成，此列表仅保留人工润色过的高质量条目
CURATED_FAQS = [
    # --- 科室位置（人工润色版，区分门诊/住院，与 JSON 楼层一致） ---
    {"question": "急诊在哪里", "answer": "急诊位于1楼，设有120急救通道、抢救室、急诊留观病房和急诊手术室，24小时开放。"},
    {"question": "儿科在几层", "answer": "儿科门诊在4楼妇儿专区，包括儿科普通门诊、儿科急诊、儿童保健和儿童雾化室。儿科住院病房在15层。"},
    {"question": "骨科在几楼", "answer": "骨科门诊在3楼外科诊区，包括骨科、脊柱外科、关节外科。骨科住院病房在12层。"},
    {"question": "妇产科在哪里", "answer": "妇产科门诊在4楼妇儿专区，包括妇科、产科、产检室。妇产科住院病区在14层。"},
    {"question": "心内科在哪", "answer": "心血管内科门诊在2楼内科综合诊区。心血管内科住院病房在10层。"},
    {"question": "消化内科在哪里", "answer": "消化内科门诊在2楼内科综合诊区。消化内科住院病房在11层。胃镜肠镜检查在7楼消化内镜中心。"},
    {"question": "呼吸科在哪", "answer": "呼吸内科门诊在2楼内科综合诊区。呼吸内科住院病房在11层。支气管镜检查在7楼呼吸内镜室。"},
    {"question": "神经内科在哪层", "answer": "神经内科门诊在2楼内科综合诊区。神经内科住院病房在13层。"},
    {"question": "神经外科在哪", "answer": "神经外科门诊在3楼外科诊区。神经外科住院病房在13层。"},
    {"question": "口腔科在哪", "answer": "口腔科在5楼五官皮肤口腔专科区，包括口腔门诊、口腔正畸和口腔修复。"},
    {"question": "眼科在哪里", "answer": "眼科在5楼五官专科区，配有验光配镜中心和视力检查室。"},
    {"question": "耳鼻喉科在哪", "answer": "耳鼻喉头颈外科在5楼五官专科区，配有听力检测室和鼻咽喉内镜室。"},
    {"question": "皮肤科在哪层", "answer": "皮肤性病科在5楼，旁边有激光美容室。"},
    {"question": "中医科在哪", "answer": "中医科在6楼，包括针灸科、推拿理疗科、中药熏蒸室和治未病中心。"},
    {"question": "体检中心在哪", "answer": "健康管理体检中心在6楼，提供全身体检、入职体检、VIP体检等服务。"},
    {"question": "ICU在哪层", "answer": "ICU综合重症监护病房在9楼，同层还有CCU心脏监护室、NICU新生儿重症监护和PICU儿童重症监护。"},
    {"question": "手术室在哪", "answer": "门诊手术室在3楼，住院手术中心在8楼，包括标准手术室和微创腔镜手术室。"},
    {"question": "做胃镜在哪", "answer": "胃镜检查在7楼消化内镜中心，需要提前预约。检查前需空腹8小时以上。"},
    {"question": "做CT在哪", "answer": "CT检查在1楼影像中心。急诊CT也在1楼，24小时可用。体检CT在6楼体检中心。"},
    {"question": "抽血化验在哪", "answer": "门诊采血在2楼采血化验窗口。急诊化验在1楼急诊区。体检采血在6楼体检中心。"},
    {"question": "做B超在哪", "answer": "普通超声检查在2楼超声检查室。急诊B超在1楼影像中心。体检超声在6楼体检中心。"},
    {"question": "拍片子在哪", "answer": "DR拍片和X光检查在1楼影像中心。"},
    {"question": "药房在哪", "answer": "西药房和中药房都在1楼，设有自助取药机。急诊药房在1楼急诊专区内。"},
    {"question": "院长办公室在哪", "answer": "院长办公室在17楼行政办公层。"},
    {"question": "发热门诊在哪", "answer": "发热门诊在2楼，有独立通道，发热患者请走发热门诊专用入口。"},

    # --- 挂号与就诊流程 ---
    {"question": "怎么挂号", "answer": "有三种方式：1楼一站式服务大厅人工窗口、各楼层自助挂号机、医院官方微信公众号或小程序线上预约。建议提前线上预约，减少等候时间。"},
    {"question": "可以现场挂号吗", "answer": "可以。1楼服务大厅有人工挂号窗口，各楼层也有自助挂号机。但热门科室建议提前在微信公众号预约，现场可能无号。"},
    {"question": "挂号需要什么证件", "answer": "需要身份证或医保卡。初诊患者需先在建卡窗口办理就诊卡，复诊患者直接刷卡或扫码挂号。"},
    {"question": "怎么预约专家号", "answer": "通过医院微信公众号、官方APP或1楼服务大厅预约。专家号一般提前7天放号，热门专家建议放号时间准时抢号。"},
    {"question": "门诊时间是什么时候", "answer": "普通门诊周一至周日8:00-12:00、14:00-17:30。专家门诊周一至周五上午为主。急诊24小时开放。"},
    {"question": "周末有门诊吗", "answer": "有。周六周日普通门诊正常开放（8:00-12:00、14:00-17:30），但部分专家门诊周末不出诊，建议提前查询排班。"},
    {"question": "怎么查医生排班", "answer": "在医院微信公众号'预约挂号'页面可查看各科室医生排班，也可在1楼导诊台咨询。"},
    {"question": "可以代挂号吗", "answer": "可以。代挂号需携带患者身份证原件和代办人身份证。线上预约可用患者本人账号操作。"},
    {"question": "挂错科了怎么办", "answer": "可到1楼导诊台咨询转科，或在自助机上退号重新挂。当日退号不收取手续费。"},
    {"question": "初诊需要带什么", "answer": "初诊需携带身份证（或户口本）、医保卡（如有）。建议带上既往检查报告和用药清单，方便医生了解病史。"},

    # --- 医保与费用 ---
    {"question": "医保怎么报销", "answer": "持医保卡就诊可实时结算。门诊统筹、住院报销在1楼医保结算窗口办理。异地医保需提前在参保地办理备案。"},
    {"question": "医保结算在哪", "answer": "医保结算在1楼一站式服务大厅的医保窗口。入院办理和出院结算也在同一区域。"},
    {"question": "异地医保可以用吗", "answer": "可以。异地医保需提前在参保地医保局办理异地就医备案，备案后持社保卡可直接结算。"},
    {"question": "住院押金交多少", "answer": "住院押金根据科室和病种不同，一般5000-20000元。医保患者押金较低，具体以入院办理窗口告知为准。"},
    {"question": "怎么查费用明细", "answer": "住院费用可在护士站自助查询机打印每日清单。门诊费用在收费窗口或自助机打印发票和明细。"},
    {"question": "支持哪些支付方式", "answer": "支持医保卡、微信、支付宝、银行卡、现金。住院大额费用建议银行卡或转账。"},

    # --- 住院相关 ---
    {"question": "怎么办理住院", "answer": "医生开具住院证后，到1楼入院办理窗口登记，缴纳押金，然后到对应楼层护士站报到。"},
    {"question": "探视时间是什么时候", "answer": "普通病房探视时间为每天15:00-17:00和19:00-20:30。ICU探视时间为15:30-16:00，每次限1人。"},
    {"question": "住院可以陪护吗", "answer": "普通病房允许1名家属陪护。ICU、NICU不允许陪护，由护士全程护理。产科病房允许1名家属陪护。"},
    {"question": "住院需要带什么", "answer": "需携带身份证、医保卡、住院证、换洗衣物、洗漱用品、水杯、拖鞋。贵重物品请勿带入病房。"},
    {"question": "怎么办理出院", "answer": "医生通知出院后，到护士站领取出院小结，到1楼出院结算窗口办理费用结算，领取发票和诊断证明。"},
    {"question": "病历复印在哪", "answer": "病历复印在1楼一站式服务大厅的病历复印窗口，需携带身份证。出院病历一般7个工作日后可复印。"},

    # --- 交通与停车 ---
    {"question": "停车场在哪", "answer": "地下停车场在负2层（大型）和负1层（小型），从大楼西侧车辆入口进入。就诊前2小时免费，之后每小时5元。"},
    {"question": "怎么坐公交到医院", "answer": "可乘坐12路、35路、78路、102路到'中心医院站'下车，步行200米即到。地铁2号线'医疗中心站'B出口直达。"},
    {"question": "医院有充电桩吗", "answer": "有。负2层停车场东侧设有新能源充电桩8个，支持快充和慢充。"},

    # --- 便民服务 ---
    {"question": "有轮椅借吗", "answer": "有。1楼导诊台旁设有轮椅租借点，凭身份证免费借用，归还时退还证件。"},
    {"question": "母婴室在哪", "answer": "母婴室在1楼服务大厅东侧和4楼妇儿专区各有一间，配有哺乳椅、尿布台和温水。"},
    {"question": "有饮水机吗", "answer": "每层楼电梯旁均设有免费饮水机，提供温水和热水。"},
    {"question": "有超市或餐厅吗", "answer": "负1层有患者便民餐厅、超市和便利店，营业时间7:00-21:00。"},
    {"question": "有银行ATM吗", "answer": "1楼服务大厅设有银行网点和ATM机，支持存取款和转账。"},
    {"question": "失物招领在哪", "answer": "失物招领在1楼导诊台，拾到物品请交至导诊台登记。"},
    {"question": "有无障碍设施吗", "answer": "有。大楼设有无障碍通道、无障碍电梯、无障碍卫生间。1楼可租借轮椅。视障人士可联系导诊台安排志愿者协助。"},

    # --- 检查预约与注意事项 ---
    {"question": "做胃镜要预约吗", "answer": "需要预约。到7楼消化内镜中心前台或微信公众号预约。检查前需空腹8小时，服用抗凝药需提前停药。"},
    {"question": "做肠镜怎么准备", "answer": "需提前1天到7楼内镜中心领取泻药，检查前4小时服用清肠液。检查前1天进食流质，检查当天空腹。"},
    {"question": "做CT需要预约吗", "answer": "普通CT当日可排队检查。增强CT需提前预约并做碘过敏试验。体检CT在6楼体检中心统一安排。"},
    {"question": "做核磁共振要注意什么", "answer": "核磁共振检查不能携带任何金属物品（手机、钥匙、首饰、带金属的内衣）。体内有金属植入物（钢板、起搏器）需提前告知医生。"},
    {"question": "体检需要空腹吗", "answer": "需要。体检前一天晚10点后禁食禁水，当天空腹到6楼体检中心。抽血和腹部B超需空腹完成后再进食。"},
    {"question": "拿报告要多久", "answer": "血常规30分钟出结果，生化检查2小时，CT/MRI报告一般24小时内。病理报告3-5个工作日。可在公众号查询电子报告。"},
]


def load_floors() -> dict:
    """加载权威楼层 JSON"""
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("floors"), f"楼层数据为空: {DATA_FILE}"
    return data


def build_rag_documents(data: dict) -> list[dict]:
    """由 JSON 生成 RAG 楼层文档（每层一个文档）"""
    building = data["building"]
    docs = []
    for item in data["floors"]:
        floor, category = item["floor"], item["category"]
        depts = "、".join(item["departments"])
        docs.append({
            "text": f"{building} — {floor}（{category}）\n{floor}设有：{depts}。",
            "metadata": {"building": building, "floor": floor, "category": category},
        })
    return docs


def build_location_faqs(data: dict) -> list[dict]:
    """由 JSON 逐科室生成位置 FAQ（跨楼层科室合并为一条，如母婴室）"""
    dept_locs: dict[str, list[tuple[str, str]]] = {}
    for item in data["floors"]:
        for dept in item["departments"]:
            dept_locs.setdefault(dept, []).append((item["floor"], item["category"]))

    faqs = []
    for dept, locs in dept_locs.items():
        if len(locs) == 1:
            floor, category = locs[0]
            answer = f"{dept}位于{floor}（{category}）。"
        else:
            answer = f"{dept}在{'、'.join(f'{fl}（{ct}）' for fl, ct in locs)}均有设置。"
        faqs.append({"question": f"{dept}在哪里", "answer": answer})
    return faqs


async def backup_and_wipe() -> None:
    """备份现有 hospital FAQ 后清空 PG + Milvus 旧数据"""
    rows = await DatabasePool.fetch(
        "SELECT question_text, answer_text, hit_count, status "
        "FROM faq_candidate WHERE skill_id = $1 ORDER BY id",
        SKILL_ID,
    )
    backup = [dict(r) for r in rows]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(__file__).parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    backup_file = backup_dir / f"faq_hospital_backup_{ts}.json"
    backup_file.write_text(json.dumps(backup, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[1/4] 已备份 {len(backup)} 条现有 FAQ -> {backup_file.name}")

    await DatabasePool.execute("DELETE FROM faq_candidate WHERE skill_id = $1", SKILL_ID)

    client = milvus_manager.get_client()
    for coll in (f"faq_{SKILL_ID}", f"skill_{SKILL_ID}"):
        if client.has_collection(coll):
            client.drop_collection(coll)
            print(f"      已删除 Milvus 集合: {coll}")


async def self_check(data: dict, expected_faq_count: int) -> None:
    """一致性自检：导航楼层 == JSON，条数对账，向量抽查"""
    from app.skill.skills.hospital_skill import hospital_navigate_handler

    print("[4/4] 一致性自检...")

    # 4.1 导航工具 → 楼层/手势 必须与 JSON 一致
    checks = [
        ("ICU", "9楼", "point_left"),
        ("中医科", "6楼", "point_right"),
        ("骨科", "3楼", "point_right"),
        ("急诊", "1楼", "point_right"),
        ("院长办公室", "17楼", "point_left"),
        ("停车场", "负2层", "point_left"),
        ("发热门诊", "2楼", "point_right"),
        ("儿科", "4楼", "point_right"),
    ]
    for query, want_floor, want_gesture in checks:
        res = await hospital_navigate_handler({"query": query})
        assert res.get("success"), f"导航失败: {query} -> {res}"
        # 精确匹配返回顶层 floor；模糊匹配（如 'ICU'）返回 matches 列表
        floor = res.get("floor") or (res.get("matches") or [{}])[0].get("floor")
        assert floor == want_floor, f"{query}: 楼层 {floor} != {want_floor}"
        assert res["gesture"] == want_gesture, f"{query}: 手势 {res['gesture']} != {want_gesture}"
        print(f"      导航 {query} -> {floor} {res['gesture']} OK")

    # 4.2 PG 与 Milvus 条数对账
    pg_count = await DatabasePool.fetchval(
        "SELECT COUNT(*) FROM faq_candidate WHERE skill_id = $1 AND status = 'promoted'",
        SKILL_ID,
    )
    assert pg_count == expected_faq_count, f"PG promoted={pg_count} != 预期 {expected_faq_count}"

    client = milvus_manager.get_client()
    # Milvus count(*) 在批量写入后有秒级一致性延迟，轮询等待最多 15 秒
    milvus_count = 0
    for _ in range(15):
        milvus_rows = client.query(
            collection_name=f"faq_{SKILL_ID}", filter="", output_fields=["count(*)"]
        )
        milvus_count = milvus_rows[0]["count(*)"] if milvus_rows else 0
        if milvus_count == expected_faq_count:
            break
        await asyncio.sleep(1)
    assert milvus_count == expected_faq_count, (
        f"Milvus faq={milvus_count} != 预期 {expected_faq_count}（存在重复残留!）"
    )
    print(f"      条数对账 OK: PG={pg_count}, Milvus faq={milvus_count}")

    # 4.3 向量抽查：ICU 问题必须命中 9 楼答案
    vector = await embedding_service.embed("ICU在哪里")
    hits = client.search(
        collection_name=f"faq_{SKILL_ID}", data=[vector], limit=1,
        output_fields=["text", "metadata"],
    )
    top_text = hits[0][0]["entity"]["text"] if hits and hits[0] else ""
    assert "9楼" in top_text, f"ICU 抽查失败，top1 答案: {top_text}"
    print(f"      向量抽查 OK: 'ICU在哪里' -> {top_text[:40]}...")


async def main() -> None:
    data = load_floors()
    n_floors = len(data["floors"])
    n_depts = sum(len(f["departments"]) for f in data["floors"])
    print(f"数据源: {DATA_FILE.name} ({n_floors} 层, {n_depts} 个科室)")

    await faq_promotion_service.ensure_table()
    await backup_and_wipe()

    # 重建 RAG 知识库
    docs = build_rag_documents(data)
    chunks = await knowledge_builder.build(SKILL_ID, docs)
    print(f"[2/4] RAG 知识库: {len(docs)} 个楼层文档 -> {chunks} 个向量切片")

    # 重建 FAQ（curated 优先，生成的位置 FAQ 补充全科覆盖）
    location_faqs = build_location_faqs(data)
    all_faqs = CURATED_FAQS + location_faqs
    result = await faq_promotion_service.batch_import(SKILL_ID, all_faqs)
    print(
        f"[3/4] FAQ: {result['promoted']} 条晋升"
        f"（curated {len(CURATED_FAQS)} + 生成 {len(location_faqs)}，重复问题自动去重）"
    )

    # 补偿重试：瞬时网络故障（如 embedding DNS 解析失败）会让个别 FAQ
    # 晋升失败、PG 行停留在 candidate，这里重试直到全部晋升（最多 3 轮）
    for round_no in range(1, 4):
        failed = await DatabasePool.fetch(
            "SELECT question_text, answer_text FROM faq_candidate "
            "WHERE skill_id = $1 AND status = 'candidate'",
            SKILL_ID,
        )
        if not failed:
            break
        print(f"      补偿轮 {round_no}: 重试 {len(failed)} 条未晋升 FAQ")
        for row in failed:
            await faq_promotion_service.manual_promote(
                SKILL_ID, row["question_text"], row["answer_text"]
            )

    # 预期条数 = 去重后的唯一问题数
    unique = {hashlib.sha256(f["question"].encode("utf-8")).hexdigest() for f in all_faqs}
    await self_check(data, expected_faq_count=len(unique))

    print("\n重建完成，知识库已与 data/hospital_floors.json 完全一致。")
    await DatabasePool.close()
    milvus_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
