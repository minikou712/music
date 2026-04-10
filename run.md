conda activate musicvis


cd C:\Users\minikou\Desktop\music


python generate_video.py --frame 70

python generate_video.py --preview --preview-seconds 30  //30s片段

新参数：--lyrics-offset（单位：秒）
正值：歌词整体提前出现一点（你现在觉得慢，就用正值）。
负值：歌词整体延后。
用法示例（生成 30 秒预览，并让歌词大约提前 0.25 秒）：

在 music 目录下执行：
python generate_video.py --preview --preview-seconds 30 --lyrics-offset 0.25


批量生成：将若干 mp3 / jpg / lrc 放到同一个文件夹，例如 songs/
命名规则：
- 音频：  歌名 - 歌手.mp3
- 封面：  歌名 - 歌手.jpg  （或 .jpeg / .png）
- 歌词：  歌名 - 歌手.lrc

示例：
cd C:\Users\minikou\Desktop\music
python batch_generate.py --input-dir songs --output-dir outputs --lyrics-offset 0.25

可选参数：
- --preview --preview-seconds 30   只导出每首歌前 30 秒
- --lyrics-offset 0.25             全局把歌词提前 0.25 秒