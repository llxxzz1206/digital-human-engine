"""生成 2D 分层形象资产 —— 精致版医院导诊员「小医」。

产出 avatar-packages/{id}/image2d/：
  base_idle.svg   底图（400x480，渐变肤色/栗色波波头/青绿制服/红十字胸牌，不含嘴）
  mouth_0..4.svg  嘴型序列（viewBox 0 0 84 50，闭合微笑 → 大张，带唇/牙/舌，clipPath 修剪）
  config.json     嘴型数量 + 嘴部区域（相对底图的比例坐标）

嘴部区域取底图坐标 (158,268,84,50)，中心 (200,293)：鼻底 ~251 之下、下巴 332 之上。
"""
from pathlib import Path
import json

AVATAR_ID = "9a46e56af580"
OUT = Path(__file__).parent / "avatar-packages" / AVATAR_ID / "image2d"
OUT.mkdir(parents=True, exist_ok=True)

# ── 底图：精致立绘，无嘴（嘴由嘴型层提供）──────────────────────────
BASE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 480" width="400" height="480">
  <defs>
    <radialGradient id="glowGrad" cx="50%" cy="42%" r="60%">
      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.16"/>
      <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="skinGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFE3CC"/>
      <stop offset="100%" stop-color="#F7C9A8"/>
    </linearGradient>
    <linearGradient id="hairGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#7A5540"/>
      <stop offset="100%" stop-color="#54382A"/>
    </linearGradient>
    <linearGradient id="uniformGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#46B0A8"/>
      <stop offset="100%" stop-color="#2E837C"/>
    </linearGradient>
    <radialGradient id="irisGrad" cx="42%" cy="38%" r="70%">
      <stop offset="0%" stop-color="#7A5540"/>
      <stop offset="100%" stop-color="#3D2B20"/>
    </radialGradient>
  </defs>

  <!-- 柔光晕 -->
  <ellipse cx="200" cy="250" rx="175" ry="215" fill="url(#glowGrad)"/>

  <!-- 后发（波波头轮廓） -->
  <path d="M 200 62 Q 96 62 80 168 Q 72 240 84 300 Q 92 348 128 374
           L 272 374 Q 308 348 316 300 Q 328 240 320 168 Q 304 62 200 62 Z"
        fill="url(#hairGrad)"/>

  <!-- 颈部 -->
  <path d="M 177 306 L 177 352 Q 200 367 223 352 L 223 306 Q 200 320 177 306 Z" fill="url(#skinGrad)"/>
  <path d="M 177 306 Q 200 322 223 306 L 223 318 Q 200 332 177 318 Z" fill="#EDB68F" opacity="0.55"/>

  <!-- 制服 -->
  <path d="M 88 480 L 88 436 Q 88 376 148 364 L 200 353 L 252 364 Q 312 376 312 436 L 312 480 Z"
        fill="url(#uniformGrad)"/>
  <!-- 衣领 -->
  <path d="M 162 360 L 200 392 L 238 360 L 230 348 L 200 372 L 170 348 Z" fill="#F5F9FB"/>
  <!-- 胸前红十字胸牌 -->
  <g transform="translate(200 430)">
    <circle r="18" fill="#F5F9FB"/>
    <rect x="-4.5" y="-12" width="9" height="24" rx="2.5" fill="#E85D5D"/>
    <rect x="-12" y="-4.5" width="24" height="9" rx="2.5" fill="#E85D5D"/>
  </g>

  <!-- 耳朵 -->
  <ellipse cx="86" cy="216" rx="15" ry="23" fill="url(#skinGrad)"/>
  <ellipse cx="314" cy="216" rx="15" ry="23" fill="url(#skinGrad)"/>
  <path d="M 82 210 Q 88 216 84 224" fill="none" stroke="#E5A47E" stroke-width="3" stroke-linecap="round"/>
  <path d="M 318 210 Q 312 216 316 224" fill="none" stroke="#E5A47E" stroke-width="3" stroke-linecap="round"/>

  <!-- 脸 -->
  <ellipse cx="200" cy="205" rx="112" ry="127" fill="url(#skinGrad)"/>

  <!-- 眉毛 -->
  <path d="M 135 175 Q 152 165 169 174" fill="none" stroke="#5C4030" stroke-width="5.5" stroke-linecap="round"/>
  <path d="M 231 174 Q 248 165 265 175" fill="none" stroke="#5C4030" stroke-width="5.5" stroke-linecap="round"/>

  <!-- 眼睛 -->
  <g>
    <circle cx="152" cy="205" r="14" fill="url(#irisGrad)"/>
    <circle cx="152" cy="205" r="6.5" fill="#241812"/>
    <circle cx="147.5" cy="200" r="4.5" fill="#FFFFFF" opacity="0.95"/>
    <circle cx="156.5" cy="209.5" r="2" fill="#FFFFFF" opacity="0.7"/>
    <path d="M 136 197 Q 152 188 168 197" fill="none" stroke="#3D2B20" stroke-width="4" stroke-linecap="round"/>
  </g>
  <g>
    <circle cx="248" cy="205" r="14" fill="url(#irisGrad)"/>
    <circle cx="248" cy="205" r="6.5" fill="#241812"/>
    <circle cx="243.5" cy="200" r="4.5" fill="#FFFFFF" opacity="0.95"/>
    <circle cx="252.5" cy="209.5" r="2" fill="#FFFFFF" opacity="0.7"/>
    <path d="M 232 197 Q 248 188 264 197" fill="none" stroke="#3D2B20" stroke-width="4" stroke-linecap="round"/>
  </g>

  <!-- 鼻子 -->
  <path d="M 197 243 Q 200 251 206 248" fill="none" stroke="#E5A47E" stroke-width="4" stroke-linecap="round"/>

  <!-- 腮红 -->
  <ellipse cx="125" cy="253" rx="21" ry="12" fill="#F2957F" opacity="0.45"/>
  <ellipse cx="275" cy="253" rx="21" ry="12" fill="#F2957F" opacity="0.45"/>

  <!-- 刘海（中分窗帘式） -->
  <path d="M 200 78 Q 108 80 96 176 Q 94 190 105 198 Q 128 150 148 138
           Q 172 126 200 152 Q 228 126 252 138 Q 272 150 295 198
           Q 306 190 304 176 Q 292 80 200 78 Z"
        fill="url(#hairGrad)"/>
  <!-- 发丝高光 -->
  <path d="M 130 108 C 155 92 245 92 270 108" fill="none" stroke="#96705A" stroke-width="7" stroke-linecap="round" opacity="0.75"/>

  <!-- 侧发束 -->
  <path d="M 90 188 Q 82 268 96 322 Q 104 344 118 342 Q 108 290 110 232 Q 111 205 105 190 Z" fill="url(#hairGrad)"/>
  <path d="M 310 188 Q 318 268 304 322 Q 296 344 282 342 Q 292 290 290 232 Q 289 205 295 190 Z" fill="url(#hairGrad)"/>

  <!-- 发夹 -->
  <g transform="translate(128 172)">
    <circle r="7.5" fill="#F0A45C"/>
    <circle r="3" fill="#FFD9A8"/>
  </g>
</svg>
"""


def mouth_svg(index: int) -> str:
    """嘴型序列：0=闭合微笑，1..4 张开程度递增（口腔+牙齿+舌头，clipPath 修剪）。"""
    if index == 0:
        # 闭合微笑（月牙形）
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 84 50" width="84" height="50">
  <path d="M 16 23.5 Q 42 31 68 23.5 Q 42 41 16 23.5 Z" fill="#D9705F"/>
</svg>
"""
    rys = [0, 6, 11, 16, 21]
    ry = rys[index]
    rx = 19 + ry * 0.28
    tw = rx * 1.35
    th = min(ry * 0.48, 9)
    tx = 42 - tw / 2
    ty = 25 - ry + 1.5
    tcy = 25 + ry * 0.42
    trx = rx * 0.52
    tr_y = ry * 0.36
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 84 50" width="84" height="50">
  <defs>
    <clipPath id="mc{index}">
      <ellipse cx="42" cy="25" rx="{rx:.2f}" ry="{ry}"/>
    </clipPath>
  </defs>
  <ellipse cx="42" cy="25" rx="{rx:.2f}" ry="{ry}" fill="#7C2F33" stroke="#D9705F" stroke-width="3"/>
  <g clip-path="url(#mc{index})">
    <rect x="{tx:.2f}" y="{ty:.2f}" width="{tw:.2f}" height="{th:.2f}" rx="3" fill="#FBF7F0"/>
    <ellipse cx="42" cy="{tcy:.2f}" rx="{trx:.2f}" ry="{tr_y:.2f}" fill="#C25650"/>
  </g>
</svg>
"""


# 嘴部区域：底图坐标 (158,268,84,50) → 比例坐标
CONFIG = {
    "mouthShapeCount": 5,
    "mouthRegion": {
        "x": round(158 / 400, 4),
        "y": round(268 / 480, 4),
        "w": round(84 / 400, 4),
        "h": round(50 / 480, 4),
    },
}

(OUT / "base_idle.svg").write_text(BASE_SVG, encoding="utf-8")
for i in range(CONFIG["mouthShapeCount"]):
    (OUT / f"mouth_{i}.svg").write_text(mouth_svg(i), encoding="utf-8")
(OUT / "config.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"OK -> {OUT}")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name}  {f.stat().st_size} bytes")
