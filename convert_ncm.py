import argparse
import ctypes
import os
from typing import Optional


# 默认假设 dll 放在工作区的 ../swap/libncmdump-1.5.1-windows-amd64/libncmdump.dll
DLL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "swap", "libncmdump-1.5.1-windows-amd64", "libncmdump.dll")
)


class NcmdumpError(Exception):
    """ncmdump 调用异常。"""


class NeteaseCryptWrapper:
    def __init__(self, dll_path: str = DLL_PATH) -> None:
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"找不到 libncmdump.dll: {dll_path}")

        self._dll = ctypes.CDLL(dll_path)

        # 函数签名：参考 src/lib/libncmdump.h
        self._dll.CreateNeteaseCrypt.argtypes = [ctypes.c_char_p]
        self._dll.CreateNeteaseCrypt.restype = ctypes.c_void_p

        self._dll.Dump.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._dll.Dump.restype = ctypes.c_int

        self._dll.FixMetadata.argtypes = [ctypes.c_void_p]
        self._dll.FixMetadata.restype = None

        self._dll.DestroyNeteaseCrypt.argtypes = [ctypes.c_void_p]
        self._dll.DestroyNeteaseCrypt.restype = None

    def convert(self, input_path: str, output_dir: Optional[str] = None) -> str:
        """将单个 .ncm 文件转换为音频，返回生成的音频文件绝对路径。

        - input_path: 源 .ncm 文件路径
        - output_dir: 输出目录，为 None 或空字符串时输出到源文件所在目录
        """
        abs_in = os.path.abspath(input_path)

        if not os.path.exists(abs_in):
            raise FileNotFoundError(f"输入文件不存在: {abs_in}")

        if not abs_in.lower().endswith(".ncm"):
            # 非 ncm 文件直接原样返回
            return abs_in

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            abs_outdir = os.path.abspath(output_dir)
            out_bytes = abs_outdir.encode("utf-8")
        else:
            abs_outdir = os.path.dirname(abs_in)
            out_bytes = b""  # 传空字符串 -> 使用原目录

        # Windows 下必须传 UTF-8 编码的文件名
        path_bytes = abs_in.encode("utf-8")

        ctx = self._dll.CreateNeteaseCrypt(path_bytes)
        if not ctx:
            raise NcmdumpError("CreateNeteaseCrypt 返回空指针")

        try:
            ret = self._dll.Dump(ctx, out_bytes)
            if ret != 0:
                raise NcmdumpError(f"Dump 失败，返回码 {ret}")

            # 写入/修复标签和封面
            self._dll.FixMetadata(ctx)
        finally:
            self._dll.DestroyNeteaseCrypt(ctx)

        # 推断输出的 mp3 或 flac 文件名
        stem = os.path.splitext(os.path.basename(abs_in))[0]
        cand_flac = os.path.join(abs_outdir, stem + ".flac")
        cand_mp3 = os.path.join(abs_outdir, stem + ".mp3")

        if os.path.exists(cand_flac):
            return os.path.abspath(cand_flac)
        if os.path.exists(cand_mp3):
            return os.path.abspath(cand_mp3)

        # 理论上不会走到这里，兜底返回 mp3 路径
        return os.path.abspath(cand_mp3)


_wrapper: Optional[NeteaseCryptWrapper] = None


def convert_ncm_to_audio(input_path: str, output_dir: Optional[str] = None) -> str:
    """模块级便捷函数：转换单个 ncm 文件。"""
    global _wrapper
    if _wrapper is None:
        _wrapper = NeteaseCryptWrapper()
    return _wrapper.convert(input_path, output_dir)


def _convert_path_or_dir(path: str, output_dir: Optional[str] = None) -> None:
    if os.path.isdir(path):
        # 遍历目录下所有 .ncm 文件
        for root, _dirs, files in os.walk(path):
            for name in files:
                if not name.lower().endswith(".ncm"):
                    continue
                in_file = os.path.join(root, name)
                print(f"转换: {in_file}")
                out_file = convert_ncm_to_audio(in_file, output_dir)
                print(f"  -> {out_file}")
    else:
        print(f"转换: {path}")
        out_file = convert_ncm_to_audio(path, output_dir)
        print(f"  -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 libncmdump.dll 将网易云 ncm 转换为音频")
    parser.add_argument("path", help="要转换的单个 ncm 文件或包含 ncm 文件的文件夹")
    parser.add_argument("--output-dir", "-o", help="输出目录，不指定则输出到源文件所在目录", default=None)
    args = parser.parse_args()

    _convert_path_or_dir(args.path, args.output_dir)


if __name__ == "__main__":
    main()
