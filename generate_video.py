import argparse
import os
import re
from typing import Tuple, List, Tuple as TupleType

import numpy as np
from moviepy.editor import AudioFileClip, VideoClip
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# 简单参数：可以按需修改
DEFAULT_TITLE = "Dear Mr 「F」"        # 默认歌曲名
DEFAULT_ARTIST = "ずっと真夜中でいいのに。"      # 默认歌手名
DEFAULT_TAG_LINES = [          # 左侧附加文案，可自由增删
    "HIFI-ON",
    "NO DELAY",
    "NO BGN",
]

DEFAULT_AUDIO_PATH = "input.mp3"  # 换成你实际的文件名
DEFAULT_COVER_PATH = "cover.jpg"   # 默认封面图
OUTPUT_PATH = "output.mp4"         # 输出视频路径
VIDEO_SIZE: Tuple[int, int] = (1920, 1080)
FPS = 30


def get_amplitude(audio: AudioFileClip, t: float) -> float:
    """返回时刻 t 附近的音量（0~1），用于驱动动画。

    使用一个很短的时间窗口从音频中截取样本，计算 RMS 音量。
    """
    window = 0.05  # 秒
    start = max(t - window / 2, 0)
    end = min(t + window / 2, audio.duration)
    if end <= start:
        return 0.0

    samples = audio.subclip(start, end).to_soundarray(fps=44100)
    if samples.size == 0:
        return 0.0
    segment = samples.astype(float)
    rms = float(np.sqrt((segment ** 2).mean()))
    amp = min(1.0, rms * 10)
    return amp


def infer_title_artist_from_path(path: str) -> Tuple[str | None, str | None]:
    """从文件名中尝试解析“歌名 - 歌手”的形式。

    例如 "I & I - Leola.mp3" -> ("I & I", "Leola").
    解析失败则返回 (None, None)。
    """
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"\s*(.+?)\s*-\s*(.+)\s*", base)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def load_lrc(path: str) -> List[TupleType[float, List[str]]]:
    """解析 LRC，返回 [(time, [line1, line2...]), ...]，按时间排序。

    支持一行多个时间标签，也支持同一时间点多行（如日文+中文翻译）。
    """
    if not os.path.exists(path):
        return []

    time_to_lines: dict[float, List[str]] = {}
    time_tag_pattern = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            tags = list(time_tag_pattern.finditer(line))
            if not tags:
                continue
            text = line[tags[-1].end():].strip()
            if not text:
                continue

            # 过滤明显的元信息行与纯音乐提示行
            meta_prefixes = ("作词", "作曲", "编曲", "编者", "制作人", "监制")
            if text.startswith(meta_prefixes):
                continue
            # 常见纯音乐提示：不当作歌词显示
            if "纯音乐" in text and "欣赏" in text:
                continue

            for m in tags:
                mmin = int(m.group(1))
                sec = float(m.group(2))
                t = mmin * 60 + sec
                bucket = time_to_lines.setdefault(t, [])
                bucket.append(text)

    items = sorted(time_to_lines.items(), key=lambda x: x[0])
    return [(float(t), lines) for t, lines in items]


def get_current_lyrics(lyrics: List[TupleType[float, List[str]]], t: float) -> List[str]:
    """给定当前时间 t，在 LRC 段中找到最后一个 time<=t 的歌词行集合。"""
    if not lyrics:
        return []
    current: List[str] = []
    for ts, lines in lyrics:
        if t + 1e-3 >= ts:
            current = lines
        else:
            break
    return current


def get_spectrum(audio: AudioFileClip, t: float, bands: int = 48) -> np.ndarray:
    """返回一个长度为 bands 的频谱强度数组（0~1）。

    使用一个短时间窗口从音频中截取样本，进行 FFT 计算频谱。
    """
    window = 0.08  # 秒
    start = max(t - window / 2, 0)
    end = min(t + window / 2, audio.duration)
    if end <= start:
        return np.zeros(bands, dtype=float)

    samples = audio.subclip(start, end).to_soundarray(fps=44100)
    if samples.size == 0:
        return np.zeros(bands, dtype=float)

    mono = samples.mean(axis=1).astype(float)
    if mono.size < 1024:
        pad = 1024 - mono.size
        mono = np.pad(mono, (pad, 0))

    N = 2048 if mono.size >= 2048 else mono.size
    window_arr = np.hanning(N)
    segment = mono[-N:] * window_arr
    spec = np.abs(np.fft.rfft(segment))[1:]

    if spec.size == 0:
        return np.zeros(bands, dtype=float)

    # log 频率分桶
    idx = np.logspace(0, np.log10(spec.size), num=bands + 1, base=10).astype(int)
    idx = np.clip(idx, 0, spec.size)
    out = np.zeros(bands, dtype=float)
    for i in range(bands):
        s = idx[i]
        e = idx[i + 1]
        if e <= s:
            out[i] = 0.0
        else:
            out[i] = float(spec[s:e].mean())

    m = out.max()
    if m > 1e-6:
        out = out / m
    # 轻微开方让能量分布更柔和
    out = np.sqrt(out)
    return out


def make_frame_factory(
    audio: AudioFileClip,
    lyrics: List[TupleType[float, List[str]]] | None,
    song_title: str,
    song_artist: str,
    tag_lines: List[str],
    cover_path: str,
    lyrics_offset: float = 0.0,
):
    """返回 moviepy 需要的 make_frame 函数。"""
    # 封面图：优先使用传入文件，无法识别或缺失时自动生成占位图，避免报错中断
    try:
        cover_img = Image.open(cover_path).convert("RGB")
    except Exception:
        cover_img = Image.new("RGB", VIDEO_SIZE, (25, 35, 60))
    w, h = VIDEO_SIZE

    # 以封面图生成一张模糊背景：先等比放大铺满，再居中裁剪并高斯模糊
    cw, ch = cover_img.size
    scale = max(w / cw, h / ch)
    bg_w = int(cw * scale)
    bg_h = int(ch * scale)
    bg = cover_img.resize((bg_w, bg_h), Image.LANCZOS)
    left = max(0, (bg_w - w) // 2)
    top = max(0, (bg_h - h) // 2)
    bg = bg.crop((left, top, left + w, top + h))
    bg = bg.filter(ImageFilter.GaussianBlur(35))
    # 稍微压暗，方便前景元素更突出
    bg = bg.point(lambda v: int(v * 0.7))
    base_bg = bg

    # 预先算好前景封面尺寸（保持比例）
    cover_target_h = int(h * 0.56)
    cover_target_w = int(cover_target_h * cw / ch)
    max_cover_w = int(w * 0.33)
    if cover_target_w > max_cover_w:
        cover_target_w = max_cover_w
        cover_target_h = int(cover_target_w * ch / cw)
    cover_main = cover_img.resize((cover_target_w, cover_target_h), Image.LANCZOS)

    # 字体在工厂中创建一次即可，避免每帧都加载
    try:
        base_dir = os.path.dirname(__file__)

        # 日文字体：项目内 dartsfont/ダーツフォント.ttf
        jp_font_path = os.path.join(base_dir, "dartsfont", "ダーツフォント.ttf")

        # 中文字体：项目内 XimSansShouXieTi/XimSansShouXieTi/Xim-Sans-ShouXieTi-2.ttf
        cn_font_path = os.path.join(base_dir, "XimSansShouXieTi", "XimSansShouXieTi", "Xim-Sans-ShouXieTi-2.ttf")

        # 顶部标题/歌手：整体用中文字体（更兼容中日韩）
        font_title = ImageFont.truetype(cn_font_path, 70)
        font_artist = ImageFont.truetype(cn_font_path, 52)

        # 备用的小号字体
        font_small = ImageFont.truetype(cn_font_path, 40)
        font_time = ImageFont.truetype(cn_font_path, 28)
        # 进度条时间用稍大一点的“伪粗体”效果
        font_time_bold = ImageFont.truetype(cn_font_path, 32)

        # 歌词：日文行用日文字体，中文行用中文字体
        font_lyric_jp_primary = ImageFont.truetype(jp_font_path, 40)
        font_lyric_jp_secondary = ImageFont.truetype(jp_font_path, 34)
        font_lyric_cn_primary = ImageFont.truetype(cn_font_path, 40)
        font_lyric_cn_secondary = ImageFont.truetype(cn_font_path, 34)
    except Exception:
        font_title = font_artist = font_small = font_time = ImageFont.load_default()
        font_time_bold = ImageFont.load_default()
        font_lyric_jp_primary = font_lyric_jp_secondary = ImageFont.load_default()
        font_lyric_cn_primary = font_lyric_cn_secondary = ImageFont.load_default()

    # 渐变与文字配色（偏冷色调，适配偏暗封面背景）
    gradient_start = (110, 200, 255)   # 左侧/起点：偏青蓝
    gradient_end = (235, 140, 255)     # 右侧/终点：偏紫粉
    text_main_color = (240, 244, 255)  # 主要文字：带一点冷色的白
    text_sub_color = (205, 215, 235)   # 次要文字：稍暗
    progress_bg_color = (150, 165, 190)

    def _lerp(a: float, b: float, t: float) -> float:
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return a + (b - a) * t

    def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        return (
            int(_lerp(c1[0], c2[0], t)),
            int(_lerp(c1[1], c2[1], t)),
            int(_lerp(c1[2], c2[2], t)),
        )

    def make_frame(t: float):
        img = base_bg.copy()
        draw = ImageDraw.Draw(img)

        # 顶部居中显示：歌曲名：xxx / 歌手：xxx
        title_text = f"歌曲名：{song_title}"
        artist_text = f"歌手：{song_artist}"
        title_y = int(h * 0.05)

        # 歌名
        bbox_title = draw.textbbox((0, 0), title_text, font=font_title)
        tw_title = bbox_title[2] - bbox_title[0]
        th_title = bbox_title[3] - bbox_title[1]
        draw.text(
            ((w - tw_title) / 2, title_y),
            title_text,
            fill=text_main_color,
            font=font_title,
        )
        # 歌手
        bbox_artist = draw.textbbox((0, 0), artist_text, font=font_artist)
        tw_artist = bbox_artist[2] - bbox_artist[0]
        # 歌手行基于歌名高度自动向下错开，避免重叠
        artist_y = title_y + th_title + 20
        draw.text(
            ((w - tw_artist) / 2, artist_y),
            artist_text,
            fill=text_sub_color,
            font=font_artist,
        )

        # 右侧清晰封面：增加圆角、浅描边与柔和阴影
        cover_x = int(w * 0.60)
        cover_y = int(h * 0.20)

        radius = int(min(cover_main.size) * 0.06)
        if radius < 10:
            radius = 10

        # 创建圆角蒙版
        mask = Image.new("L", cover_main.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle(
            [0, 0, cover_main.size[0] - 1, cover_main.size[1] - 1],
            radius=radius,
            fill=255,
        )

        # 柔和阴影（稍向右下偏移），使用冷色渐变中间色 + 半透明蒙版，避免过黑
        shadow_offset = 14
        shadow_color = _lerp_color(gradient_start, gradient_end, 0.4)
        shadow = Image.new("RGB", cover_main.size, shadow_color)
        shadow = shadow.filter(ImageFilter.GaussianBlur(22))
        # 将圆角蒙版整体压低到约 30% 不透明度，使阴影更轻
        shadow_mask = mask.point(lambda v: int(v * 0.15))
        img.paste(
            shadow,
            (cover_x + shadow_offset, cover_y + shadow_offset),
            shadow_mask,
        )

        # 细白描边矩形，提升层次感
        border_margin = 4
        border_rect = [
            cover_x - border_margin,
            cover_y - border_margin,
            cover_x + cover_main.size[0] + border_margin,
            cover_y + cover_main.size[1] + border_margin,
        ]
        draw.rounded_rectangle(
            border_rect,
            radius=radius + border_margin,
            outline=(245, 245, 250),
            width=2,
        )

        # 实际圆角封面
        img.paste(cover_main, (cover_x, cover_y), mask)

        # 左侧区域现在全部留给滚动歌词使用
        left_x = int(w * 0.07)
        top_y = int(h * 0.24)

        # 计算当前音量和频谱
        amp = get_amplitude(audio, t)
        spectrum = get_spectrum(audio, t, bands=64)

        # 底部整宽频谱（竖线），类似截图风格
        base_y = int(h * 0.92)
        max_bar_h = int(h * 0.30)
        n = spectrum.size
        if n > 0:
            x_start = int(w * 0.03)
            x_end = int(w * 0.97)
            step = (x_end - x_start) / n
            for i, v in enumerate(spectrum):
                strength = float(v) * (0.35 + 0.9 * amp)
                bar_h = max(6, max_bar_h * strength)
                x = x_start + step * i
                # 根据横向位置使用冷色渐变，并随能量调整亮度
                t_pos = i / (n - 1) if n > 1 else 0.0
                base_c = _lerp_color(gradient_start, gradient_end, t_pos)
                factor = 0.4 + 0.6 * min(1.0, strength * 1.2)
                color = (
                    int(base_c[0] * factor),
                    int(base_c[1] * factor),
                    int(base_c[2] * factor),
                )
                draw.line((x, base_y, x, base_y - bar_h), fill=color, width=3)

        # 左侧滚动歌词：上一段 + 当前段，每段（日文+中文）成对显示
        if lyrics:
            # 根据偏移调整“歌词时间”，正值表示整体提前显示一些
            eff_t = t + lyrics_offset

            # 找到当前所在的歌词段索引
            idx_cur = -1
            for i, (ts, _lines) in enumerate(lyrics):
                if eff_t + 1e-3 >= ts:
                    idx_cur = i
                else:
                    break

            if idx_cur >= 0:
                def split_lines(ls: List[str]) -> tuple[str, str | None]:
                    """将一段歌词拆分为 (jp_line, cn_line)。

                    - 两行：按约定第一行为日文，第二行为中文。
                    - 一行：自动判断是否包含假名；
                      * 含平假名/片假名 → 认为是日文行；
                      * 否则 → 认为是纯中文行。
                    """
                    if not ls:
                        return "", None

                    if len(ls) == 1:
                        text = ls[0]
                        # 检测是否包含日文假名（平假名/片假名范围）
                        has_kana = any(
                            "\u3040" <= ch <= "\u30ff" for ch in text
                        )
                        if has_kana:
                            return text, None  # 纯日文
                        else:
                            return "", text  # 纯中文

                    # len >= 2：第一行为日文，第二行为中文
                    return ls[0], ls[1]

                cur_ts, cur_lines = lyrics[idx_cur]
                cur_jp, cur_cn = split_lines(cur_lines)

                if idx_cur - 1 >= 0:
                    prev_jp, prev_cn = split_lines(lyrics[idx_cur - 1][1])
                else:
                    prev_jp, prev_cn = None, None

                # 固定上下位置：上一段固定在上方，当前段固定在中间，避免上下“反复横跳”
                base_y_lyric = int(h * 0.50)
                offset = 110
                y_cur_center = base_y_lyric
                y_prev_center = base_y_lyric - offset

                # 左对齐，宽度控制在封面左侧区域
                lyric_x = left_x

                line_gap = 8

                # 上一段（淡一点），日文在上、中文在下
                if prev_jp or prev_cn:
                    block_lines: list[tuple[str, ImageFont.FreeTypeFont]] = []
                    if prev_jp:
                        # 日文上一段：用日文字体
                        block_lines.append((prev_jp, font_lyric_jp_secondary))
                    if prev_cn:
                        # 中文上一段：用中文字体
                        block_lines.append((prev_cn, font_lyric_cn_secondary))

                    heights = []
                    for text, f in block_lines:
                        bb = draw.textbbox((0, 0), text, font=f)
                        heights.append(bb[3] - bb[1])
                    total_h = sum(heights) + line_gap * (len(block_lines) - 1)
                    y_start = y_prev_center - total_h / 2
                    cur_y = y_start
                    for (text, f), line_h in zip(block_lines, heights):
                        draw.text(
                            (lyric_x, cur_y),
                            text,
                            fill=text_sub_color,
                            font=f,
                        )
                        cur_y += line_h + line_gap

                # 当前段（主高亮），日文在上、中文在下
                if cur_jp or cur_cn:
                    block_lines_cur: list[tuple[str, ImageFont.FreeTypeFont, tuple[int, int, int]]] = []
                    if cur_jp:
                        # 当前日文行：用日文字体 + 渐变主色
                        block_lines_cur.append((cur_jp, font_lyric_jp_primary, gradient_start))
                    if cur_cn:
                        # 当前中文行：用中文字体 + 略偏暖的次主色
                        blend_c = _lerp_color(gradient_start, gradient_end, 0.55)
                        block_lines_cur.append((cur_cn, font_lyric_cn_primary, blend_c))

                    heights_cur = []
                    for text, f, _c in block_lines_cur:
                        bb = draw.textbbox((0, 0), text, font=f)
                        heights_cur.append(bb[3] - bb[1])
                    total_h_cur = sum(heights_cur) + line_gap * (len(block_lines_cur) - 1)
                    y_start_cur = y_cur_center - total_h_cur / 2
                    cur_y = y_start_cur
                    shadow_offset = 2
                    for (text, f, color), line_h in zip(block_lines_cur, heights_cur):
                        # 轻微阴影
                        draw.text(
                            (lyric_x + shadow_offset, cur_y + shadow_offset),
                            text,
                            fill=(0, 0, 0),
                            font=f,
                        )
                        draw.text(
                            (lyric_x, cur_y),
                            text,
                            fill=color,
                            font=f,
                        )
                        cur_y += line_h + line_gap

        # 频谱下方的进度条（居中细线 + 圆形滑块 + 扬声器图标）
        if audio.duration > 0:
            progress = max(0.0, min(1.0, t / audio.duration))
            bar_y = int(h * 0.965)
            bar_x_start = int(w * 0.08)
            bar_x_end = int(w * 0.86)  # 给右侧扬声器预留空间

            # 背景轨道
            draw.line(
                (bar_x_start, bar_y, bar_x_end, bar_y),
                fill=progress_bg_color,
                width=6,
            )

            # 已播放部分
            cur_x = bar_x_start + (bar_x_end - bar_x_start) * progress
            bar_color = _lerp_color(gradient_start, gradient_end, progress)
            draw.line(
                (bar_x_start, bar_y, cur_x, bar_y),
                fill=bar_color,
                width=8,
            )

            # 圆形滑块
            knob_r = 12
            draw.ellipse(
                [cur_x - knob_r, bar_y - knob_r, cur_x + knob_r, bar_y + knob_r],
                fill=bar_color,
            )

            # 右侧简易扬声器图标
            spk_x = bar_x_end + 24
            spk_y = bar_y
            # 小矩形 + 三角
            body_w, body_h = 10, 14
            draw.rectangle(
                [spk_x, spk_y - body_h // 2, spk_x + body_w, spk_y + body_h // 2],
                fill=bar_color,
            )
            draw.polygon(
                [
                    (spk_x + body_w, spk_y - body_h // 2),
                    (spk_x + body_w + 10, spk_y),
                    (spk_x + body_w, spk_y + body_h // 2),
                ],
                fill=bar_color,
            )

            # 右侧时间显示（当前播放时间，格式 mm:ss）
            cur_sec = max(0.0, min(float(audio.duration), float(t)))
            mm = int(cur_sec) // 60
            ss = int(cur_sec) % 60
            time_str = f"{mm:02d}:{ss:02d}"
            bbox_time = draw.textbbox((0, 0), time_str, font=font_time_bold)
            tw_time = bbox_time[2] - bbox_time[0]
            th_time = bbox_time[3] - bbox_time[1]
            time_x = spk_x + body_w + 24
            # 略高于进度条中线，让文字稍微往上提一点
            time_y = bar_y - th_time / 2 - 6
            draw.text(
                (time_x, time_y),
                time_str,
                fill=text_sub_color,
                font=font_time_bold,
            )

        return np.array(img)

    return make_frame


def main():
    parser = argparse.ArgumentParser(description="Generate music visualization video.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="仅渲染前几秒预览视频 (preview_output.mp4)",
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=10.0,
        help="预览视频时长，单位秒，默认 10 秒",
    )
    parser.add_argument(
        "--frame",
        type=float,
        default=None,
        help="只输出指定时间点的一帧 PNG，用于快速看布局，单位：秒",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="音频文件路径，可选；若不提供则尝试用 '歌名 - 歌手.mp3'",
    )
    parser.add_argument(
        "--lyrics",
        type=str,
        default=None,
        help="歌词 LRC 路径，可选；若不提供则尝试用 '歌名 - 歌手.lrc'",
    )
    parser.add_argument(
        "--cover",
        type=str,
        default=None,
        help="封面图片路径，可选，默认 cover.jpg",
    )
    parser.add_argument("--title", type=str, default=None, help="歌曲名")
    parser.add_argument("--artist", type=str, default=None, help="歌手名")
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=None,
        help="左侧附加文案，多项用空格分隔",
    )
    parser.add_argument(
        "--lyrics-offset",
        type=float,
        default=0.0,
        help="歌词整体时间偏移（秒），正值表示歌词提前出现",
    )
    args = parser.parse_args()

    # 推断音频路径
    if args.audio:
        audio_path = args.audio
    else:
        audio_path = DEFAULT_AUDIO_PATH

    # 尝试从音频文件名解析“歌名 - 歌手”
    inferred_title, inferred_artist = infer_title_artist_from_path(audio_path)

    # 最终歌曲名/歌手名优先级：命令行参数 > 文件名推断 > 默认值
    song_title = args.title or inferred_title or DEFAULT_TITLE
    song_artist = args.artist or inferred_artist or DEFAULT_ARTIST

    # 推断歌词路径：优先命令行，其次与音频同名 .lrc
    if args.lyrics:
        lyrics_path = args.lyrics
    else:
        base_no_ext = os.path.splitext(audio_path)[0]
        lyrics_path = base_no_ext + ".lrc"

    cover_path = args.cover or DEFAULT_COVER_PATH

    # 打开音频
    audio = AudioFileClip(audio_path)

    # 加载歌词（如果存在）
    lyrics_segments: List[TupleType[float, List[str]]] | None = None
    if os.path.exists(lyrics_path):
        lyrics_segments = load_lrc(lyrics_path)

    lyrics_offset = float(args.lyrics_offset)

    # 左侧标签
    tag_lines = args.tags if args.tags else DEFAULT_TAG_LINES

    # 只渲染单帧图片的快速预览模式
    if args.frame is not None:
        t = max(0.0, min(float(args.frame), max(0.0, audio.duration - 1e-3)))
        make_frame = make_frame_factory(
            audio,
            lyrics_segments,
            song_title,
            song_artist,
            tag_lines,
            cover_path,
            lyrics_offset=lyrics_offset,
        )
        frame = make_frame(t)
        img = Image.fromarray(frame)
        img.save("preview_frame.png")
        print(f"Saved preview_frame.png at t={t:.2f}s")
        audio.close()
        return

    make_frame = make_frame_factory(
        audio,
        lyrics_segments,
        song_title,
        song_artist,
        tag_lines,
        cover_path,
        lyrics_offset=lyrics_offset,
    )

    duration = audio.duration
    output_path = OUTPUT_PATH
    if args.preview:
        # 预览时使用可配置时长，但不超过音频总长
        preview_len = max(0.1, float(args.preview_seconds))
        duration = min(preview_len, audio.duration)
        output_path = "preview_" + OUTPUT_PATH

    clip = VideoClip(make_frame, duration=duration)
    clip = clip.set_audio(audio.subclip(0, duration))
    clip.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate="6000k",
    )

    # 主动关闭资源，避免 Windows 上 moviepy 析构时偶发的句柄警告
    clip.close()
    audio.close()


if __name__ == "__main__":
    main()
