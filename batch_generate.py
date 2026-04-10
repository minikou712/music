import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

from moviepy.editor import AudioFileClip, VideoClip
from PIL import Image

from generate_video import (
    load_lrc,
    make_frame_factory,
    infer_title_artist_from_path,
    DEFAULT_TAG_LINES,
    DEFAULT_COVER_PATH,
    FPS,
)


def find_cover_for_audio(audio_path: Path) -> Path:
    """在音频同目录下根据文件名寻找封面图，优先同名 jpg/png/jpeg。"""
    base = audio_path.with_suffix("")
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    # 找不到同名封面则返回默认 cover.jpg（在项目根目录）
    return Path(DEFAULT_COVER_PATH)


def find_lyrics_for_audio(audio_path: Path) -> Optional[Path]:
    """在音频同目录下根据文件名寻找 lrc 歌词。"""
    cand = audio_path.with_suffix(".lrc")
    return cand if cand.exists() else None


def render_one(
    audio_path: Path,
    output_path: Path,
    *,
    lyrics_path: Optional[Path] = None,
    cover_path: Optional[Path] = None,
    preview: bool = False,
    preview_seconds: float = 10.0,
    lyrics_offset: float = 0.0,
) -> None:
    """对单个音频生成可视化视频。"""
    audio_path = audio_path.resolve()
    output_path = output_path.resolve()

    if not cover_path:
        cover_path = find_cover_for_audio(audio_path)
    if not lyrics_path:
        lyrics_path = find_lyrics_for_audio(audio_path)

    audio = AudioFileClip(str(audio_path))

    lyrics_segments: Optional[List[Tuple[float, List[str]]]] = None
    if lyrics_path and lyrics_path.exists():
        lyrics_segments = load_lrc(str(lyrics_path))

    inferred_title, inferred_artist = infer_title_artist_from_path(str(audio_path))
    song_title = inferred_title or audio_path.stem
    song_artist = inferred_artist or ""

    make_frame = make_frame_factory(
        audio,
        lyrics_segments,
        song_title,
        song_artist,
        DEFAULT_TAG_LINES,
        str(cover_path),
        lyrics_offset=lyrics_offset,
    )

    duration = audio.duration
    if preview:
        preview_len = max(0.1, float(preview_seconds))
        duration = min(preview_len, audio.duration)

    clip = VideoClip(make_frame, duration=duration)
    clip = clip.set_audio(audio.subclip(0, duration))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip.write_videofile(
        str(output_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate="6000k",
    )

    clip.close()
    audio.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate music visualization videos.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="输入目录，扫描其中的 .mp3 文件",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="输出目录，默认 ./outputs",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="仅渲染预览片段而非整首",
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=10.0,
        help="预览片段时长，单位秒，默认 10 秒",
    )
    parser.add_argument(
        "--lyrics-offset",
        type=float,
        default=0.0,
        help="歌词整体时间偏移（秒），正值表示歌词提前",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在或不是目录: {input_dir}")

    mp3_files = sorted(input_dir.glob("*.mp3"))
    if not mp3_files:
        raise SystemExit(f"在目录中没有找到 mp3 文件: {input_dir}")

    print(f"发现 {len(mp3_files)} 个音频文件，将批量生成视频……")

    for audio_path in mp3_files:
        out_name = audio_path.stem + ".mp4"
        out_path = output_dir / out_name
        print(f"[生成] {audio_path.name} -> {out_path}")
        render_one(
            audio_path,
            out_path,
            lyrics_path=find_lyrics_for_audio(audio_path),
            cover_path=find_cover_for_audio(audio_path),
            preview=args.preview,
            preview_seconds=args.preview_seconds,
            lyrics_offset=args.lyrics_offset,
        )


if __name__ == "__main__":
    main()
