"""把 AI 立绘处理成 2D 分层形象资产（精致动漫风）。

步骤：
  1. 抹掉立绘上画出来的嘴（按列在唇区做上下肤色渐变插值 + 高斯柔化）→ base_idle.png
  2. 生成与立绘风格匹配的嘴型序列 mouth_0..4.png（140x90，闭合微笑→大张）
  3. 写 config.json（mouthRegion 对齐检测到的实际唇位）

唇部检测结论（本立绘 1024x1280）：唇 x 449..575（中心 512），y 766..792（中心 ~779）。
"""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFilter

SRC = Path(r"C:/Users/lxz/.qoderworkcn/workspace/mrq46j1dtwevx1p9/vibe_images/xiaoyi-base_1784617874.png")
AVATAR_ID = "9a46e56af580"
OUT = Path(__file__).parent / "avatar-packages" / AVATAR_ID / "image2d"
OUT.mkdir(parents=True, exist_ok=True)

# ── 几何参数（源图 1024x1280 坐标系）──
# 抹除范围需盖住唇本身(766-792)+人中阴影(~742起)+唇下褶皱(~819止)，否则残留红棕印记
ERASE_X0, ERASE_X1 = 440, 584      # 抹嘴矩形（略宽于唇，保证抹净）
ERASE_Y0, ERASE_Y1 = 744, 816      # 抹嘴矩形纵向（含人中阴影与唇下褶皱）
ABOVE_BAND = (728, 738)            # 上缘取样带（鼻下干净肤色，避开人中阴影）
BELOW_BAND = (824, 834)            # 下缘取样带（下巴干净肤色，避开唇下褶皱）

# 嘴型层区域（源图坐标）→ 比例坐标
REGION_X0, REGION_Y0 = 442, 735
REGION_W, REGION_H = 140, 90
SRC_W, SRC_H = 1024, 1280

CONFIG = {
    "mouthShapeCount": 5,
    "mouthRegion": {
        "x": round(REGION_X0 / SRC_W, 4),
        "y": round(REGION_Y0 / SRC_H, 4),
        "w": round(REGION_W / SRC_W, 4),
        "h": round(REGION_H / SRC_H, 4),
    },
}


def erase_mouth(im: Image.Image) -> Image.Image:
    """按列做上下肤色线性插值，填补唇区，再柔化边缘。"""
    im = im.convert("RGB")
    px = im.load()

    def band_avg(x: int, band: tuple[int, int]) -> tuple[float, float, float]:
        ys = range(band[0], band[1] + 1)
        n = len(ys)
        r = sum(px[x, y][0] for y in ys) / n
        g = sum(px[x, y][1] for y in ys) / n
        b = sum(px[x, y][2] for y in ys) / n
        return r, g, b

    span = ERASE_Y1 - ERASE_Y0
    for x in range(ERASE_X0, ERASE_X1 + 1):
        top = band_avg(x, ABOVE_BAND)
        bot = band_avg(x, BELOW_BAND)
        for y in range(ERASE_Y0, ERASE_Y1 + 1):
            t = (y - ERASE_Y0) / span
            px[x, y] = (
                int(top[0] + (bot[0] - top[0]) * t),
                int(top[1] + (bot[1] - top[1]) * t),
                int(top[2] + (bot[2] - top[2]) * t),
            )

    # 柔化填补区边缘，避免色块感
    pad = 6
    box = (ERASE_X0 - pad, ERASE_Y0 - pad, ERASE_X1 + pad, ERASE_Y1 + pad)
    region = im.crop(box).filter(ImageFilter.GaussianBlur(2.2))
    im.paste(region, (box[0], box[1]))
    return im


# ── 嘴型序列（140x90，与 mouthRegion 同宽高比，中心 (70,45)）──
LIP_LINE = (176, 84, 68)       # 闭合唇线（深玫瑰）
LIP_BODY = (206, 112, 94)      # 唇色
MOUTH_IN = (108, 48, 54)       # 口腔
TEETH = (251, 247, 241)
TONGUE = (199, 94, 86)
CX, CY = 70, 45


def _soft(img: Image.Image, radius: float = 1.1) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def _quad(p0, c, p2, n=28):
    """二次贝塞尔曲线采样，返回点列。"""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t ** 2 * p2[1]
        pts.append((x, y))
    return pts


def make_mouth(index: int) -> Image.Image:
    W, H = REGION_W, REGION_H
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if index == 0:
        # 闭合微笑：贝塞尔围出唇形（温和上扬），填唇色 + 上唇轮廓 + 下唇高光
        lc, rc = (14, 38), (126, 38)          # 嘴角（略高于唇中线 → 上扬微笑）
        top = _quad(lc, (70, 54), rc)          # 上唇线，中心约 y=46
        bot = _quad(rc, (70, 70), lc)          # 下唇底，中心约 y=54
        d.polygon(top + bot, fill=LIP_BODY + (255,))
        # 上唇轮廓（深玫瑰，闭合线）
        d.line(top, fill=LIP_LINE + (255,), width=3)
        # 下唇受光高光
        d.ellipse([50, 50, 90, 59], fill=(246, 202, 186, 165))
        # 嘴角轻阴影
        d.ellipse([8, 34, 18, 43], fill=(150, 70, 58, 80))
        d.ellipse([122, 34, 132, 43], fill=(150, 70, 58, 80))
        return _soft(img, 1.0)

    # 张开嘴型：口腔(深) + 唇缘 + 牙齿 + 舌头，尺寸随 index 递增
    inner_ry = [0, 9, 15, 21, 27][index]
    inner_rx = [0, 26, 31, 35, 38][index]
    ix0, iy0 = CX - inner_rx, CY - inner_ry
    ix1, iy1 = CX + inner_rx, CY + inner_ry

    # 唇缘（比口腔略大一圈）
    d.ellipse([ix0 - 4, iy0 - 4, ix1 + 4, iy1 + 4], fill=LIP_BODY + (255,))
    # 口腔
    d.ellipse([ix0, iy0, ix1, iy1], fill=MOUTH_IN + (255,))

    # 牙齿 / 舌头：画在独立层，用口腔椭圆做遮罩，保证不溢出
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).ellipse([ix0, iy0, ix1, iy1], fill=255)

    inner = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    di = ImageDraw.Draw(inner)
    # 牙齿：上排，宽度随嘴宽
    tw = inner_rx * 1.5
    th = min(inner_ry * 0.55, 11)
    di.rounded_rectangle([CX - tw / 2, iy0 + 1, CX + tw / 2, iy0 + 1 + th], radius=4, fill=TEETH + (255,))
    # 舌头：下方
    if inner_ry >= 14:
        di.ellipse([CX - inner_rx * 0.55, CY + inner_ry * 0.18, CX + inner_rx * 0.55, iy1 + 2], fill=TONGUE + (255,))
    inner.putalpha(mask)
    img = Image.alpha_composite(img, inner)

    # 上唇高光
    d2 = ImageDraw.Draw(img)
    d2.arc([ix0 + 6, iy0 - 6, ix1 - 6, iy0 + inner_ry], start=200, end=340, fill=(235, 170, 150, 120), width=3)
    return _soft(img, 1.0)


def main() -> None:
    src = Image.open(SRC)

    # 1. 抹嘴 → 底图
    base = erase_mouth(src)
    base.save(OUT / "base_idle.png", "PNG")

    # 2. 嘴型序列
    for i in range(CONFIG["mouthShapeCount"]):
        make_mouth(i).save(OUT / f"mouth_{i}.png", "PNG")

    # 3. config
    (OUT / "config.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")

    # 调试：嘴部区域裁剪（抹嘴前后对比）
    dbg = Image.new("RGB", (REGION_W * 2 + 8, REGION_H), (30, 30, 30))
    dbg.paste(src.convert("RGB").crop((REGION_X0, REGION_Y0, REGION_X0 + REGION_W, REGION_Y0 + REGION_H)), (0, 0))
    dbg.paste(base.crop((REGION_X0, REGION_Y0, REGION_X0 + REGION_W, REGION_Y0 + REGION_H)), (REGION_W + 8, 0))
    dbg.save(OUT.parent / "_mouth_erase_debug.png")

    print(f"OK -> {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}  {f.stat().st_size} bytes")
    print("mouthRegion:", CONFIG["mouthRegion"])


if __name__ == "__main__":
    main()
