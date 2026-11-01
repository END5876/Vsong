#!/usr/bin/env python3
"""
vsong.py — 從直播、錄影或錄音中自動擷取「只有歌」的段落

流程：yt-dlp 下載音訊 → AST 音訊分類模型（GPU）每 3 秒判斷「唱歌 / 說話」
      → 平滑、合併 → ffmpeg 切成單曲 .m4a + 播放清單（電腦、手機都能播）

────────────────────────────────────────
安裝（Windows 11 + NVIDIA 顯卡）
  1. 安裝 ffmpeg：  winget install Gyan.FFmpeg   （裝完重開終端機）
  2. 安裝 PyTorch（CUDA 版，指令依 https://pytorch.org 選擇，例如）：
       pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  3. pip install -U yt-dlp transformers numpy soundfile
  4.（選用，辨識歌名）pip install shazamio

  ※ 若 YouTube 下載失敗：先 pip install -U yt-dlp，並依 yt-dlp 說明安裝 Deno（JS runtime）。

────────────────────────────────────────
用法
  python vsong.py "https://www.youtube.com/watch?v=XXXX"
  python vsong.py URL1 URL2 ...              # 一次處理多場
  python vsong.py URL --identify             # 用 Shazam 嘗試辨識歌名
  python vsong.py URL --merge                # 另外輸出一個「全部歌曲串在一起」的檔案
  python vsong.py URL --threshold 0.6        # 調整判斷門檻（用快取分數，幾秒就重算完）
  python vsong.py URL --cookies-from-browser chrome   # 會員限定影片

  播放清單 / 頻道（自動展開成每部影片，已處理過的會跳過）
  python vsong.py "https://www.youtube.com/playlist?list=XXXX"
  python vsong.py "https://www.youtube.com/@頻道名/streams" --filter 歌枠,karaoke,歌ってみた --limit 20
  python vsong.py 清單網址 --force           # 已處理過的也重做

  電腦裡的檔案（影片或音檔都可以，資料夾會找出裡面所有影音檔）
  python vsong.py "D:/錄影/karaoke.mp4" --channel "我的頻道"
  python vsong.py "D:/錄影/Aちゃん" --channel "Aちゃん Ch" --filter 歌枠

  手動修正：編輯 out/<頻道>/<影片ID>/timeline.json 的 start / end / title，然後
  python vsong.py --recut out/<頻道>/<影片ID>
  python vsong.py --library                  # 只重建各頻道的總歌單

輸出（每個頻道一個資料夾）
  out/<頻道>/<影片ID>/songs/01 0h12m34s.m4a ...  ← 單曲檔
  out/<頻道>/<影片ID>/songs/playlist.m3u8        ← 這一場的播放清單
  out/<頻道>/<影片ID>/timeline.json              ← 時間軸（可手動編輯）
  out/<頻道>/<影片ID>/scores.csv                 ← 每 3 秒的分數，調參用
  out/<頻道>/all_songs.m3u8                      ← 這位的所有歌（依直播日期排序）
  out/<頻道>/versions/<歌名>.m3u8                ← 同一首歌在不同場的所有版本（需有歌名）
  out/<頻道>/library.csv                         ← 歌庫總表（Excel 可開，含 YouTube 時間連結）
  out/channels.json                              ← 頻道 ID ↔ 資料夾名稱對照（頻道改名也不會分家）

  頻道以「上傳者」為準：聯動直播會放在開台那位的資料夾。
  同步到手機：同步整個 out（或單一頻道資料夾），排除 source.* 與 scores.*（原始音訊很大）；
  播放清單用的是相對路徑，手機上也能直接用。
  舊版的 out/<影片ID>/ 會在下次執行時自動搬進對應的頻道資料夾。
"""

import argparse
import asyncio
import base64
import contextlib
import csv
import datetime
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np

__version__ = "0.1.0"

SR = 16000          # 分析用取樣率
WIN = 6.0           # 每個分析視窗長度（秒）
HOP = 3.0           # 視窗間距（秒）＝時間解析度
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

# AudioSet 標籤（找不到的會自動略過）
SING_LABELS = ["Singing", "Male singing", "Female singing", "Vocal music",
               "Song", "Choir", "A capella"]
MUSIC_LABELS = ["Music"]
SPEECH_LABELS = ["Speech", "Male speech, man speaking", "Female speech, woman speaking",
                 "Child speech, kid speaking", "Conversation", "Narration, monologue"]

# 偵測：講話分數低於 TALK 算「沒人講話」；比 SILENT_DB 小聲算無聲
TALK = 0.3
SILENT_DB = -70.0
QUIET_GAP = 30.0      # 兩段之間沒人講話、不超過這麼多秒 → 可能是間奏，合併成同一首
MAX_SONG = 390.0      # 合併後超過 6.5 分鐘就不合併（比較可能是不講話直接接下一首）
SHORT_PART = 150.0
TALK_SMOOTH = 15      # 「持續在講話」：講話分數取前後 15 個點（約 45 秒）的中位數；歌裡 10 幾秒的 rap、口白不算
SPLIT_LEN = 390.0     # 比這個長的段落可能是好幾首連著唱，在音量突然掉下去的地方切開
SPLIT_DROP = 6.0      # 比這段的中位數小聲幾 dB 算「歌和歌之間的空檔」    # 而且其中一段要短於 2.5 分鐘（一首歌被切成兩半時，通常有一半很短）

# 連 YouTube 的間隔（--sleep 設定）；好幾部同時處理時也共用，真的要連線時才等
NET_SLEEP = (5.0, 10.0)
# 原曲長度（iTunes Search API，免費、不用金鑰，每分鐘約 20 次）：只查短於這個秒數的歌（可能被切掉或是 TV size）
LENGTH_CHECK = 180.0
ITUNES_GAP = 3.0

# Holodex（holodex.net）：社群整理的歌曲標記（有開始和結束時間）和時間戳留言。金鑰寫在 .env 的 HOLODEX_API_KEY
HOLODEX_API = "https://holodex.net/api/v2"
HOLODEX_GAP = 1.0          # 每次請求至少隔幾秒

# 依粉絲歌單切：沒寫結束時間、又沒偵測到這首時，先給幾秒；一首最多切幾秒
SETLIST_GUESS = 240.0
SETLIST_MAX_SONG = 600.0
MERGE_MUSIC_MIN = 0.35   # 間奏裡有講話時，伴奏分數一直在這以上、
MERGE_MUSIC_MEAN = 0.45  # 平均在這以上、
MERGE_DROP = 10.0        # 音量（中位數）沒比前面小這麼多 dB，
MUSIC_GAP = 45.0         # 間隔不超過這麼多秒，就當成同一首（あくび 11 場手動切的：漏掉的歌 3 → 0）
OUTRO_MUSIC = 0.45    # 尾奏（前奏）：唱完後（開唱前）伴奏分數還在這以上、
OUTRO_DROP = 8.0      # 音量沒比這首小這麼多 dB，就算還在尾奏（即使在講話）
OUTRO_MAX = 45.0      # 尾奏最多往後延伸幾秒
ONSET_BACK = 30.0     # 粉絲歌單的開始時間最多往前調幾秒（找前奏真正開始的地方）
ONSET_GAP_DB = 20.0   # 比這首小這麼多 dB 算「安靜」（伴奏音軌開始前的空隙）
ONSET_MARGIN = 0.5    # 在伴奏開始前多留幾秒
END_BACK = 8.0        # 結尾細調：在目前結尾前幾秒～
END_FWD = 8.0         # 後幾秒內，找伴奏衰減完、
END_GAP_DB = 25.0     # 比這首小這麼多 dB 的安靜，
END_MARGIN = 0.8      # 再多留幾秒（餘音）；
END_MAX_EARLY = 2.0   # 最多只往前調幾秒（寧可多留伴奏）。手動切的 19 場：結尾 2 秒內 67% → 73%，太早 11% → 7%
SETLIST_CHAIN_GAP = 60.0  # 依歌單切時，間奏、口白後面隔不到這麼多秒又唱起來的，算同一首（不超過下一首）

# 歌名辨識（Gemini）：每首送一段音訊，32 tokens/秒，40 秒約 1300 tokens
GEMINI_MODEL = "gemini-3.1-flash-lite"
ENV_FILE = Path(__file__).resolve().parent / ".env"   # GEMINI_API_KEY=... 寫在這裡
GEMINI_CLIP = 40.0
GEMINI_TIMEOUT = 60.0        # 每次請求最多等幾秒（搜尋 1.5 倍）
GEMINI_WORKERS = 4           # 同時辨識幾首（時間都花在等 Google 回應；免費方案有每分鐘次數上限，不要開太多）
CUT_WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))  # 同時切幾首（每個 ffmpeg 約用一個核心，留一半給別的程式）
GEMINI_CLIPS = (0.45, 0.7)   # 第一段認不出來時，換歌曲 70% 的位置再試一次
# Gemini 憑記憶猜歌名幾乎都錯（而且很有自信），所以只請它「聽寫歌詞」，歌名一律用歌詞去 Google 搜尋
LISTEN_PROMPT = """這是直播或演出中一段翻唱歌曲的錄音（大多是日文歌）。
請把聽到的歌詞盡量完整、正確地寫下來，照原本的語言。不要猜歌名。
只輸出一個 JSON 物件，不要任何其他文字：
{"lyrics": "聽到的歌詞", "singing": true 或 false（有沒有人在唱歌）}"""
SEARCH_PROMPT = """以下是從一首翻唱歌曲聽寫下來的歌詞（可能有錯字或漏字）：
{lyrics}
請用 Google 搜尋其中最有特色的一兩句歌詞，找出原曲。最多搜尋 3 次。
找到候選後，確認那首歌的歌詞裡真的有上面聽到的句子（允許聽寫的錯字）；確認不了就填 null，不要猜。
只輸出一個 JSON 物件，不要任何其他文字：
{{"title": "原曲歌名（原文）或 null", "artist": "原唱或 null", "matched": "原曲歌詞中對上的那一句", "confidence": 0 到 1 的數字}}"""

# 電腦裡的檔案
LOCAL_CHANNEL = "本機檔案"   # 沒指定 --channel 時放這個資料夾
MEDIA_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".ts", ".m2ts", ".wmv", ".m4v",
              ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma"}
# 音軌編碼 → 不重新編碼、直接複製時用的副檔名；其他編碼會轉成 AAC
AUDIO_COPY = {"aac": "m4a", "alac": "m4a", "mp3": "mp3", "opus": "opus", "vorbis": "ogg", "flac": "flac"}


# ───────────────────────── 工具 ─────────────────────────
def ffmpeg(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


# 好幾部影片同時處理時共用的資源
_net_lock, _net_last = threading.RLock(), [0.0]
_fs_lock = threading.Lock()              # channels.json 這類共用檔案
_gpu_lock = threading.Lock()             # AI 分析一次一部（模型共用、顯示記憶體不打架）
_sems, _sems_lock = {}, threading.Lock()


def _sem(name, n):
    """同名的工作（ffmpeg、Gemini）不管幾部影片同時處理，總數都不超過 n"""
    with _sems_lock:
        return _sems.setdefault((name, n), threading.BoundedSemaphore(max(1, n)))


@contextlib.contextmanager
def youtube_slot():
    """連 YouTube 的一次請求（讀資訊 + 下載、讀留言、展開清單）都包在這裡面：
    一次只連一個；跟上一次請求「結束」至少隔 NET_SLEEP 秒（避免被當成機器人）。
    好幾部同時處理時大家排同一個隊；音訊已經下載好、不用連線的就不用排"""
    lo, hi = NET_SLEEP
    with _net_lock:
        if hi > 0 and _net_last[0]:
            wait = _net_last[0] + random.uniform(lo, hi) - time.monotonic()
            if wait > 0:
                print(f"  等 {wait:.0f} 秒再連 YouTube（避免被當成機器人）", flush=True)
                time.sleep(wait)
        try:
            yield
        finally:
            _net_last[0] = time.monotonic()


def youtube_wait():
    with youtube_slot():
        pass


class TaggedOutput:
    """好幾部同時處理時，每行前面標上是哪一部（例如「[12/62] 」）。
    進度列（\r 覆寫的那種）只留最後一次，換行時才印，避免好幾部的進度混在同一行"""

    def __init__(self, raw):
        self.raw, self.lock, self.local = raw, threading.Lock(), threading.local()

    def set_tag(self, tag):
        self.local.tag, self.local.buf = tag, ""

    def end(self):
        if getattr(self.local, "buf", ""):
            self.write("\n")
        self.local.tag = None

    def write(self, text):
        tag = getattr(self.local, "tag", None)
        if tag is None:
            with self.lock:
                self.raw.write(text)
            return len(text)
        buf = self.local.buf
        for part in re.split(r"(\r|\n)", text):
            if part == "\r":
                buf = ""
            elif part == "\n":
                if buf.strip():
                    with self.lock:
                        self.raw.write(f"{tag}{buf.strip()}\n")
                        self.raw.flush()
                buf = ""
            else:
                buf += part
        self.local.buf = buf
        return len(text)

    def flush(self):
        with self.lock:
            self.raw.flush()

    def __getattr__(self, name):
        return getattr(self.raw, name)


def keep_tag(fn):
    """ThreadPoolExecutor 裡的工作沿用呼叫者那一部影片的標籤（[12/62]），一行一次輸出，不會跟別部混在一起"""
    out = sys.stdout
    tag = getattr(out.local, "tag", None) if isinstance(out, TaggedOutput) else None
    if tag is None:
        return fn

    def run(*a, **k):
        out.set_tag(tag)
        try:
            return fn(*a, **k)
        finally:
            out.end()
    return run


def youtube_id(url):
    u = str(url)
    if not re.match(r"https?://(?:[\w-]+\.)?(?:youtube\.com|youtube-nocookie\.com|youtu\.be)/", u):
        return None                      # 其他網站（例如 OPENREC）照舊交給 yt-dlp
    m = re.search(r"(?:[?&]v=|youtu\.be/|/shorts/|/live/|/embed/)([\w-]{11})(?![\w-])", u)
    return m.group(1) if m else None


def fmt_t(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def safe_name(name):
    return re.sub(r"[\\/:*?\"<>|']+", "_", name).strip()[:80]


def song_filename(n, s):
    label = s["title"] if s.get("title") else fmt_t(float(s["start"]))
    return safe_name(f"{n:02d} {label}") + ".m4a"


def fmt_date(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if d and len(d) == 8 else (d or "")


def norm_title(t):
    """比對重複歌名用：全形半形統一、去空白標點、不分大小寫"""
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", t)).lower()


def local_id(path):
    """本機檔案的「影片 ID」：檔名 + 大小，檔案搬位置也認得出是同一個"""
    h = hashlib.sha1(f"{path.name}|{path.stat().st_size}".encode("utf-8")).hexdigest()[:10]
    return f"file-{h}"


def list_media(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTS)


# ───────────────────────── 0. 展開播放清單 / 頻道 / 資料夾 ─────────────────────────
def cookie_opts(cookies):
    """yt-dlp 的 cookie 設定。cookies 可以是 cookies.txt 的路徑，或瀏覽器名稱（chrome / edge / firefox）"""
    if not cookies:
        return {}
    if str(cookies).lower().endswith(".txt") or Path(cookies).is_file():
        return {"cookiefile": str(cookies)}
    return {"cookiesfrombrowser": (cookies,)}


def _playlists_path(cache_dir):
    return Path(cache_dir) / "playlists.json" if cache_dir else None


def load_playlist(cache_dir, url):
    p = _playlists_path(cache_dir)
    try:
        return json.loads(p.read_text(encoding="utf-8")).get(url) if p else None
    except Exception:
        return None


def save_playlist(cache_dir, url, title, items):
    p = _playlists_path(cache_dir)
    if not p or not items:
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data[url] = {"title": title, "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "items": [list(x) for x in items]}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def expand_urls(urls, cookies=None, keywords=None, limit=None, cache_dir=None):
    """回傳 [(url 或檔案路徑, video_id, 來自清單?)]。播放清單、頻道、資料夾會展開成各部影片。
    展開過的清單存在 cache_dir/playlists.json：之後讀不到 YouTube（cookies 過期、被擋）就用上次的清單"""
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "extract_flat": "in_playlist"}
    opts.update(cookie_opts(cookies))

    jobs, seen = [], set()

    def add(url, vid, from_list):
        if vid and vid in seen:
            return
        seen.add(vid)
        jobs.append((url, vid, from_list))

    def walk(ydl, info, depth, found):
        for e in info.get("entries") or []:
            if not e:
                continue
            if e.get("_type") == "playlist" or e.get("ie_key") == "YoutubeTab":
                # 頻道首頁會先給「影片 / 直播」等分頁，再往下展開一層
                if depth < 2:
                    if e.get("entries"):
                        sub = e
                    else:
                        with youtube_slot():
                            sub = ydl.extract_info(e.get("url") or e.get("webpage_url"), download=False)
                    walk(ydl, sub, depth + 1, found)
                continue
            if e.get("live_status") in ("is_upcoming", "is_live"):
                continue
            vid = e.get("id")
            if vid:
                found.append((f"https://www.youtube.com/watch?v={vid}", vid, e.get("title") or ""))

    with contextlib.ExitStack() as stack:
        ydl = None
        for u in urls:
            p = Path(u)
            if p.is_dir():
                found = [f for f in list_media(p)
                         if not keywords or any(k.lower() in f.stem.lower() for k in keywords)]
                if limit:
                    found = found[:limit]
                print(f"▶ 資料夾「{p.name}」找到 {len(found)} 個影音檔")
                for f in found:
                    add(str(f), local_id(f), True)
                continue
            if p.is_file():
                add(str(p), local_id(p), False)
                continue
            vid = youtube_id(u)
            if vid and "list=" not in u:         # 單一影片：網址裡就有 ID，不用先連 YouTube
                add(f"https://www.youtube.com/watch?v={vid}", vid, False)
                continue
            if ydl is None:                      # 只有真的要連 YouTube 才載入 yt-dlp
                import yt_dlp
                ydl = stack.enter_context(yt_dlp.YoutubeDL(opts))
            try:
                with youtube_slot():
                    info = ydl.extract_info(u, download=False)
                found = None
                if info.get("_type") == "playlist" or info.get("entries") is not None:
                    found = []
                    walk(ydl, info, 0, found)
                    save_playlist(cache_dir, u, info.get("title"), found)
            except Exception as ex:
                saved = load_playlist(cache_dir, u)
                if not saved:
                    print(f"✖ 無法讀取 {u}：{ex}")
                    continue
                print(f"⚠ 讀不到 YouTube（{str(ex).splitlines()[0][:80]}），改用 {saved['time']} 存下的清單")
                info, found = {"title": saved["title"]}, [tuple(x) for x in saved["items"]]
            if found is not None:
                if keywords:
                    found = [x for x in found if any(k.lower() in x[2].lower() for k in keywords)]
                if limit:
                    found = found[:limit]
                print(f"▶ 清單「{info.get('title')}」展開為 {len(found)} 部影片")
                for url, vid, _ in found:
                    add(url, vid, True)
            else:
                add(u, info.get("id"), False)
    return jobs


# ───────────────────────── 頻道資料夾 ─────────────────────────
def channel_dir(out_root, channel_id, channel_name):
    """依頻道 ID 決定資料夾；第一次看到的名稱就固定下來，之後頻道改名也沿用同一個資料夾"""
    with _fs_lock:                       # 好幾部同時處理時，channels.json 不能同時寫
        return _channel_dir(out_root, channel_id, channel_name)


def _channel_dir(out_root, channel_id, channel_name):
    idx_p = out_root / "channels.json"
    try:
        idx = json.loads(idx_p.read_text(encoding="utf-8"))
    except Exception:
        idx = {}
    key = channel_id or channel_name or "unknown"
    changed = False
    if key not in idx and channel_id and channel_name in idx:
        idx[key] = idx.pop(channel_name)          # 舊資料只記了名稱，補上 ID
        changed = True
    if key not in idx:
        name = safe_name(channel_name or key).rstrip(". ") or key
        if name in idx.values():                 # 不同頻道撞名
            name = f"{name}_{key[-6:]}"
        idx[key] = name
        changed = True
    if changed:
        out_root.mkdir(parents=True, exist_ok=True)
        idx_p.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_root / idx[key]


def find_video_dir(out_root, vid):
    hits = list(out_root.glob(f"*/{vid}/timeline.json"))
    return hits[0].parent if hits else None


def migrate_old_layout(out_root):
    """舊版輸出 out/<影片ID>/ → out/<頻道>/<影片ID>/"""
    for tl_p in list(out_root.glob("*/timeline.json")):
        d = tl_p.parent
        try:
            tl = json.loads(tl_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        dest = channel_dir(out_root, tl.get("channel_id"), tl.get("channel")) / d.name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(dest))
        print(f"  搬移舊資料：{d.name} → {dest.parent.name}/")
    for name in ("all_songs.m3u8", "library.csv"):
        (out_root / name).unlink(missing_ok=True)
    if (out_root / "versions").is_dir():
        shutil.rmtree(out_root / "versions")


# ───────────────────────── 1. 下載 ─────────────────────────
META_KEYS = ("video_id", "title", "channel", "channel_id", "url", "upload_date")


def cached_audio(out_root, vid):
    """已經下載好的音訊：回傳 (資料夾, 影片資訊或 None, 音訊檔名)，沒有就 None。
    影片資訊存在 timeline.json（處理完的）或 meta.json（下載好、還沒處理完的）裡"""
    for d in sorted({p.parent for p in Path(out_root).glob(f"*/{glob.escape(vid)}/source.*")}):
        srcs = [p for p in d.glob("source.*") if p.suffix not in (".part", ".ytdl") and ".tmp." not in p.name]
        if not srcs:
            continue
        for name in ("timeline.json", "meta.json"):
            try:
                info = json.loads((d / name).read_text(encoding="utf-8"))
            except Exception:
                continue
            if info.get("title") and not info.get("meta_partial"):
                src = info.get("source") if info.get("source") and (d / info["source"]).exists() else srcs[0].name
                return d, {k: info.get(k) for k in META_KEYS}, src
        return d, None, srcs[0].name
    return None


def download(url, out_root, cookies=None):
    # 音訊已經下載好、影片資訊也存著：不用再連 YouTube（cookies 過期、被擋時照樣能處理）
    vid0 = youtube_id(url)
    hit = cached_audio(out_root, vid0) if vid0 else None
    if hit and hit[1]:
        d0, meta, src = hit
        print(f"  頻道：{meta['channel']} → {d0.parent.name}/")
        print(f"  已有音訊快取：{src}（不用連 YouTube）")
        return d0, {**meta, "source": src}

    import yt_dlp

    base = {"quiet": True, "no_warnings": True, "noplaylist": True}
    base.update(cookie_opts(cookies))

    with youtube_slot():                   # 讀資訊和下載算同一次連線，一起排隊
        try:
            with yt_dlp.YoutubeDL({**base, "skip_download": True}) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            if not hit:
                raise
            # 舊版下載的（沒存影片資訊）：讀不到 YouTube 就先用資料夾名稱，標題之後重新處理時會補上
            d0, _, src = hit
            print(f"  ⚠ 讀不到 YouTube（{str(e).splitlines()[0][:80]}），改用已下載的音訊：{src}；影片標題、日期暫時不明")
            return d0, {"video_id": vid0, "title": vid0, "channel": d0.parent.name, "channel_id": None,
                        "url": f"https://www.youtube.com/watch?v={vid0}", "upload_date": None, "source": src,
                        "meta_partial": True}

        vid = info["id"]
        channel = info.get("channel") or info.get("uploader")
        channel_id = info.get("channel_id") or info.get("uploader_id")
        d = (hit[0] if hit else None) or find_video_dir(out_root, vid) or channel_dir(out_root, channel_id, channel) / vid
        d.mkdir(parents=True, exist_ok=True)
        print(f"  頻道：{channel} → {d.parent.name}/")
        meta = {
            "video_id": vid,
            "title": info.get("title"),
            "channel": channel,
            "channel_id": channel_id,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "upload_date": info.get("upload_date"),
        }
        # 先存影片資訊：下載完但後面分析、切檔中斷的，下次不用再連 YouTube
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        existing = [p for p in d.glob("source.*") if p.suffix not in (".part", ".ytdl")]
        if existing:
            src = existing[0]
            print(f"  已有音訊快取：{src.name}")
        else:
            print("  下載音訊中…")
            opts = {**base, "format": "bestaudio/best", "outtmpl": str(d / "source.%(ext)s")}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info2 = ydl.extract_info(url, download=True)
                src = Path(ydl.prepare_filename(info2))
    meta["source"] = src.name
    return d, meta


def audio_codec(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=codec_name", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return ""            # 沒有 ffprobe 或讀不到：交給 ffmpeg 轉檔


def extract_audio(path, d):
    """從影片 / 音檔取出第一條音軌存成 source.*；AAC、MP3、Opus 等直接複製，不重新編碼"""
    codec = audio_codec(path)
    if codec is None:
        raise RuntimeError("這個檔案沒有音軌")
    ext = AUDIO_COPY.get(codec)
    if ext:
        tmp = d / f"source.tmp.{ext}"
        try:
            ffmpeg(["-i", str(path), "-map", "0:a:0", "-vn", "-c:a", "copy", str(tmp)])
        except subprocess.CalledProcessError:
            tmp.unlink(missing_ok=True)
            ext = None
    if not ext:
        ext = "m4a"
        tmp = d / f"source.tmp.{ext}"
        ffmpeg(["-i", str(path), "-map", "0:a:0", "-vn", "-c:a", "aac", "-b:a", "256k", str(tmp)])
    src = d / f"source.{ext}"
    tmp.replace(src)
    return src


def import_local(path, out_root, channel=None):
    """電腦裡的影片 / 音檔：取出音軌放進 out/<頻道>/<file-xxxx>/，之後流程和 YouTube 一樣"""
    path = Path(path).resolve()
    vid = local_id(path)
    channel = channel or LOCAL_CHANNEL
    d = find_video_dir(out_root, vid) or channel_dir(out_root, None, channel) / vid
    d.mkdir(parents=True, exist_ok=True)
    print(f"  本機檔案：{path.name} → {d.parent.name}/")
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    meta = {
        "video_id": vid,
        "title": path.stem,
        "channel": channel,
        "channel_id": None,
        "url": None,
        "local_path": str(path),
        "upload_date": mtime.strftime("%Y%m%d"),
    }
    existing = [p for p in d.glob("source.*") if ".tmp." not in p.name]
    if existing:
        src = existing[0]
        print(f"  已有音訊快取：{src.name}")
    else:
        print("  取出音軌中…")
        src = extract_audio(path, d)
    meta["source"] = src.name
    return d, meta


# ───────────────────────── 2. 分析 ─────────────────────────
_ast, _ast_lock = {}, threading.Lock()


def analyze(wav_path, batch=16):
    import soundfile as sf
    import torch
    from transformers import ASTForAudioClassification, AutoFeatureExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  ⚠ 沒偵測到 CUDA，改用 CPU（會很慢）。請確認裝的是 CUDA 版 PyTorch。")

    with _ast_lock:
        if device not in _ast:
            _ast[device] = (AutoFeatureExtractor.from_pretrained(MODEL_ID),
                            ASTForAudioClassification.from_pretrained(MODEL_ID).to(device).eval())
        fe, model = _ast[device]
    name2id = {v: int(k) for k, v in model.config.id2label.items()}

    def ids(names):
        return [name2id[n] for n in names if n in name2id]

    groups = [ids(SING_LABELS), ids(MUSIC_LABELS), ids(SPEECH_LABELS)]

    win, hop = int(WIN * SR), int(HOP * SR)
    with sf.SoundFile(str(wav_path)) as f:
        total = f.frames
        n = max(1, (total - win) // hop + 1)
        out = np.zeros((n, 3), dtype=np.float32)
        buf, idxs = [], []
        for i in range(n):
            f.seek(i * hop)
            x = f.read(win, dtype="float32")
            if len(x) < win:
                x = np.pad(x, (0, win - len(x)))
            buf.append(x)
            idxs.append(i)
            if len(buf) == batch or i == n - 1:
                inp = fe(buf, sampling_rate=SR, return_tensors="pt")["input_values"].to(device)
                with torch.no_grad(), torch.autocast(device_type=device, enabled=(device == "cuda")):
                    probs = torch.sigmoid(model(input_values=inp).logits).float().cpu().numpy()
                for c, g in enumerate(groups):
                    if g:
                        out[idxs, c] = probs[:, g].max(axis=1)
                buf, idxs = [], []
                print(f"\r  分析中 {i + 1}/{n}（{(i + 1) / n:.0%}）", end="", flush=True)
    print()
    return out


def loudness(src, n):
    """每個分析點（和 AST 同樣的 6 秒視窗、3 秒間距）的音量（dB）"""
    # 邊解碼邊算每 3 秒的能量，不把整場音訊放進記憶體（6 小時的直播也只要幾 KB）
    sr = 8000
    hop = int(HOP * sr)
    per = int(round(WIN / HOP))                          # 一個視窗 = 2 個間距
    energy = np.zeros(n + per)
    count = np.zeros(n + per)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn",
           "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    k = 0
    with proc.stdout as f:
        while k < len(energy):
            data = f.read(hop * 2)
            if len(data) < 2:
                break
            x = np.frombuffer(data[:len(data) // 2 * 2], np.int16).astype(np.float64) / 32768
            energy[k], count[k] = np.dot(x, x), len(x)
            k += 1
    if proc.poll() is None:                              # 已經算夠了，剩下的不用解碼
        proc.kill()
    proc.wait()
    if not count.any():
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    e = sum(energy[j:j + n] for j in range(per))
    c = sum(count[j:j + n] for j in range(per))
    return (10 * np.log10(e / np.maximum(c, 1) + 1e-12)).astype(np.float32)


def loud_threshold(loud):
    """把整場的音量分成「聊天」和「唱歌」兩群的分界（Otsu）。完全無聲的片段不算"""
    v = loud[loud > SILENT_DB]
    if len(v) < 10 or np.ptp(v) < 6:              # 整場音量都差不多：音量沒有參考價值
        return float(np.median(v if len(v) else loud))
    hist, edges = np.histogram(v, 100)
    c = (edges[:-1] + edges[1:]) / 2
    w1 = np.cumsum(hist)
    w2 = w1[-1] - w1
    m1 = np.cumsum(hist * c)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m1[-1] * w1 / w1[-1] - m1) ** 2 / (w1 * w2)
    return float(c[np.nanargmax(between[:-1])])


def _median(x, w):
    xp = np.pad(x, w // 2, mode="edge")
    return np.array([np.median(xp[i:i + w]) for i in range(len(x))])


def score_of(scores, loud=None):
    """唱歌時 music、sing 高、speech 低；邊放 BGM 邊聊天時 speech 會把分數拉下來。
    AST 對「卡拉 OK 伴奏 + 人聲」常常認不出 singing，所以再加上音量：唱歌通常比聊天大聲很多。
    但持續在講話時（大聲聊天、邊放 BGM 邊講）音量只能扣分不能加分，不然講話會被當成歌、或接在歌後面"""
    sing, music, speech = scores[:, 0], scores[:, 1], scores[:, 2]
    s = music + sing - speech
    if loud is not None:
        lt = np.clip((loud[:len(s)] - loud_threshold(loud)) / 10, -1, 1)
        talking = _median(speech, TALK_SMOOTH) >= 0.5      # 歌裡短短一句喊聲不算
        s = s + np.where(talking, np.minimum(lt, 0), lt)
    return s


def _runs(on):
    """連續為 True 的區段 → [[第一點, 最後一點], ...]"""
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    return runs


def detect(scores, threshold=0.5, min_len=60.0, max_gap=8.0, smooth=5, loud=None, quiet_gap=QUIET_GAP):
    s = score_of(scores, loud)
    pad = smooth // 2
    sp = np.pad(s, pad, mode="edge")
    s = np.array([np.median(sp[i:i + smooth]) for i in range(len(s))])
    speech = scores[:, 2]

    merged = []        # [第一點, 最後一點, 最後一塊（間奏之後那段）的第一點]
    for i, j in _runs(s >= threshold):
        if merged:
            pi, pj, piece = merged[-1]
            gap = (i - pj - 1) * HOP
            if gap <= max_gap:
                merged[-1][1] = j
                continue
            # 間奏、安靜的段落會掉分。中間沒人講話、合起來也不像兩首歌，就當成同一首。
            # 一首歌可能有好幾段間奏；比的是間奏前後「那兩塊」，避免一路串起好幾首短歌
            # 有在講話、但伴奏沒斷、音量也沒掉太多的（間奏裡講幾句話），也是同一首
            quiet = speech[pj + 1:i].mean() < TALK
            music_on = False
            if not quiet and loud is not None:
                ref = float(np.median(loud[max(0, pj - 20):pj + 1]))
                mus = scores[pj + 1:i, 1]
                music_on = mus.min() >= MERGE_MUSIC_MIN and mus.mean() >= MERGE_MUSIC_MEAN \
                    and float(np.median(loud[pj + 1:i])) >= ref - MERGE_DROP
            fits = (j - pi + 1) * HOP <= MAX_SONG and min(pj - piece + 1, j - i + 1) * HOP <= SHORT_PART
            if (gap <= quiet_gap and quiet or gap <= MUSIC_GAP and music_on) and fits:
                merged[-1] = [pi, j, i]
                continue
        merged.append([i, j, i])
    segs = [(max(0.0, i * HOP + WIN / 2 - HOP / 2), j * HOP + WIN / 2 + HOP / 2) for i, j, _ in merged]
    return [(a, b) for a, b in segs if b - a >= min_len]


def widen(segs, scores, threshold=0.5, pad=2.0, extend=15.0, loud=None):
    """歌曲前後容易被切掉（安靜的前奏、清唱開頭、尾音還在響），所以：
    1. 邊界往外延伸：沒人講話、或分數還在「門檻 − 0.5」以上就繼續，最多 extend 秒（碰到無聲就停）
    2. 再前後各多留 pad 秒
    不會超出音檔，也不會和前後首重疊（重疊時切在中間）"""
    s = score_of(scores, loud)
    speech = scores[:, 2]
    n = len(s)
    total = (n - 1) * HOP + WIN
    lo = threshold - 0.5

    def keep(k):
        if loud is not None and loud[k] <= SILENT_DB:
            return False
        return speech[k] < TALK or s[k] >= lo

    out = []
    for a, b in segs:
        i = int(round((a + HOP / 2 - WIN / 2) / HOP))    # 這首的第一個點
        j = int(round((b - HOP / 2 - WIN / 2) / HOP))    # 最後一個點
        a2, b2 = a, b
        while i - 1 >= 0 and keep(i - 1) and a - (a2 - HOP) <= extend:
            i, a2 = i - 1, a2 - HOP
        while j + 1 < n and keep(j + 1) and (b2 + HOP) - b <= extend:
            j, b2 = j + 1, b2 + HOP
        out.append([max(0.0, a2 - pad), min(total, b2 + pad)])
    for k in range(1, len(out)):
        if out[k - 1][1] > out[k][0]:
            # 切在兩邊放寬後的中間，但不能切進任何一首原本偵測到的範圍
            lo_cut, hi_cut = segs[k - 1][1], segs[k][0]
            mid = (out[k - 1][1] + out[k][0]) / 2
            mid = min(max(mid, lo_cut), hi_cut) if lo_cut <= hi_cut else (lo_cut + hi_cut) / 2
            out[k - 1][1], out[k][0] = mid, mid
    return [(round(a, 1), round(b, 1)) for a, b in out if b > a]


def split_long(segs, loud, max_len=SPLIT_LEN, drop=SPLIT_DROP, min_piece=100.0):
    """太長的段落（好幾首連著唱、中間沒講話）在音量突然掉下去的地方切開，每段至少 min_piece 秒"""
    if loud is None:
        return list(segs)
    out = []

    def frame(t):
        return int(round((t - WIN / 2) / HOP))

    def go(a, b):
        if b - a <= max_len:
            out.append((a, b))
            return
        i, j = max(0, frame(a)), min(frame(b), len(loud) - 1)
        med = float(np.median(loud[i:j + 1]))
        best = None
        for k in range(i, j + 1):
            t = k * HOP + WIN / 2
            if t - a >= min_piece and b - t >= min_piece and loud[k] <= med - drop \
                    and (best is None or loud[k] < loud[best]):
                best = k
        if best is None:
            out.append((a, b))
            return
        t = round(best * HOP + WIN / 2, 1)
        go(a, t)
        go(t, b)

    for a, b in segs:
        go(a, b)
    return out


def trim_talk_tail(segs, scores, keep=2.0):
    """結尾接著一直講話的，切到最後一個沒在講話的點再多留 keep 秒"""
    speech = scores[:, 2]
    out = []
    for a, b in segs:
        i = int(round((a + HOP / 2 - WIN / 2) / HOP))
        j = min(int(round((b - HOP / 2 - WIN / 2) / HOP)), len(speech) - 1)
        k = j
        while k > i and speech[k] >= 0.5:
            k -= 1
        out.append((a, min(b, round(k * HOP + WIN / 2 + HOP / 2 + keep, 1))) if k < j else (a, b))
    return out


def extend_outro(segs, scores, loud, music_min=OUTRO_MUSIC, drop=OUTRO_DROP, max_ext=OUTRO_MAX, trim=2.0, fixed=()):
    """唱完、開始講話時伴奏常常還在（尾奏）：伴奏分數還高、音量沒比這首小 drop dB 以上，就繼續往後，
    即使在講話也算，最多 max_ext 秒、不超過下一首；切在最後一個「還有伴奏」的點的中間附近。
    fixed：這幾段（索引）的結尾不動（手動輸入的歌單寫了結束時間）
    （あくび 11 場、205 首手動切的：結尾 5 秒內 68% → 82%，太早 36% → 15%）"""
    if loud is None:
        return list(segs)
    music = scores[:, 1]
    n = min(len(music), len(loud))
    total = (len(scores) - 1) * HOP + WIN
    out = []
    for k, (a, b) in enumerate(segs):
        lim = segs[k + 1][0] if k + 1 < len(segs) else total
        i0 = max(0, int((a - WIN / 2) // HOP))
        i1 = min(n - 1, int((b - WIN / 2) // HOP))
        if i1 < i0 or k in fixed:
            out.append((a, b))
            continue
        med = float(np.median(loud[i0:i1 + 1]))
        j, last = i1 + 1, None
        while j < n and j * HOP + WIN / 2 - b <= max_ext and music[j] >= music_min and loud[j] >= med - drop:
            last, j = j, j + 1
        if last is not None:
            b = max(b, min(lim, round(last * HOP + WIN / 2 + HOP / 2 - trim, 1)))
        out.append((a, b))
    return out


def fine_loudness(src, a, b, sr=8000, step=0.05):
    """src 的 a～b 秒：每 step 秒的音量（dB），細調邊界用"""
    a = max(0.0, a)
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}",
                          "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    n = int(step * sr)
    m = len(x) // n
    return a + np.arange(m) * step, 10 * np.log10((x[:m * n].reshape(m, n) ** 2).mean(1) + 1e-10)


def setlist_onset(src, scores, loud, start, end, lo, back=ONSET_BACK, gap_db=ONSET_GAP_DB, gap_len=0.3,
                  margin=ONSET_MARGIN):
    """粉絲歌單的開始時間有人標得晚（前奏切掉一段）：伴奏音軌開始前通常有一段很短的安靜，
    歌單時間落在連續的伴奏中間時，往前找最近的安靜，從安靜結束前一點點開始切。
    只往前、不往後：歌單時間那裡還沒有伴奏的（清唱起頭、標得早）照歌單；最遠到 lo（上一首的結尾）"""
    lo = max(lo, start - back)
    if start - lo < 1.0:
        return start
    try:
        t, db = fine_loudness(src, lo, start + 8)
    except (OSError, subprocess.CalledProcessError):
        return start
    after = db[(t >= start) & (t < start + 8)]
    if not len(after):
        return start
    quiet = db < np.percentile(after, 90) - gap_db
    need, run, ends = int(round(gap_len / 0.05)), 0, []
    for i, q in enumerate(quiet):                  # 每段安靜（至少 gap_len 秒）結束的時間
        run = run + 1 if q else 0
        if run >= need and (i + 1 == len(quiet) or not quiet[i + 1]):
            ends.append(float(t[i]) + 0.05)
    near = [e for e in ends if start - 1.0 <= e <= start + 1.5]
    if near:                                       # 歌單時間就在安靜之後：對的（晚了一點點就往前對齊）
        return round(max(lo, min(start, min(near) - margin)), 1)
    before = [e for e in ends if e < start - 1.0]
    if not before:
        return start
    e = max(before)
    i0, i1 = int((start - WIN / 2) // HOP), int((end - WIN / 2) // HOP)
    med = float(np.median(loud[max(0, i0):i1 + 1]))
    frames = range(int(e // HOP), int(round((start - WIN / 2) / HOP)) + 1)
    if all(0 <= j < len(scores) and scores[j, 1] >= OUTRO_MUSIC and loud[j] >= med - OUTRO_DROP for j in frames):
        return round(max(lo, e - margin), 1)       # 安靜之後到歌單時間一直是伴奏：前奏從那裡開始
    return start


def align_end(src, a, b, nxt, back=END_BACK, fwd=END_FWD, gap_db=END_GAP_DB, gap_len=0.15, margin=END_MARGIN,
              max_early=END_MAX_EARLY):
    """結尾用每 0.05 秒的音量對齊：伴奏最後一個音衰減完、聲音降到很低（比這首小 gap_db）的那一刻，再多留 margin 秒。
    在目前結尾前 back 秒～後 fwd 秒內找第一段這樣的安靜；最多只往前 max_early 秒（寧可多留伴奏），不超過下一首"""
    hi = min(nxt, b + fwd)
    if hi - max(a + 30, b - 15) < 2:
        return b
    try:
        t, db = fine_loudness(src, b - 20, b + 8)
    except (OSError, subprocess.CalledProcessError):
        return b
    ref = db[(t >= b - 15) & (t < b - 3)]                  # 這首結尾前的音量
    if not len(ref):
        return b
    quiet = db < np.percentile(ref, 90) - gap_db
    need, run = int(round(gap_len / 0.05)), 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run == need and b - back <= t[i - need + 1] <= hi:
            nb = float(t[i - need + 1]) + margin
            return round(min(max(nb, a + 30, b - max_early), nxt), 1)
    return b


def quiet_runs(t, db, thr, min_len):
    """db 低於 thr、連續至少 min_len 秒的段落 [(開始, 結束)]"""
    out, i, need = [], 0, int(round(min_len / 0.05))
    q = db < thr
    while i < len(q):
        if not q[i]:
            i += 1
            continue
        j = i
        while j < len(q) and q[j]:
            j += 1
        if j - i >= need:
            out.append((float(t[i]), float(t[j - 1]) + 0.05))
        i = j
    return out


# 切點檢查：網頁上標出「程式沒把握」的切點，讓人優先確認
CHECKS = {
    "end": ("結尾可疑", "結尾附近沒有伴奏結束後的安靜：可能切在伴奏中間，或後面接著 BGM／講話"),
    "start": ("開頭可疑", "開頭附近沒有伴奏開始前的安靜：可能切在前奏中間，或是清唱、邊講話邊開始的歌"),
    "gap": ("後段沒伴奏", "歌的後段有一段沒有伴奏：歌可能早就結束了，後面接到別的段落"),
    "long": ("比原曲長很多", "比原曲長 45 秒以上：可能接到別的段落"),
}


def check_song(src, scores, a, b, orig_len=None):
    """依現在的切點檢查音訊，回傳沒把握的項目（CHECKS 的 key）。
    手動切的 19 場：標出來的約 17%，其中 7 成真的差 2 秒以上；差 10 秒以上的抓到一半"""
    out = []
    try:
        t, db = fine_loudness(src, b - 20, b + 8)
        ref = db[(t >= b - 15) & (t < b - 3)]
        if len(ref) and not any(b - 2.0 <= s <= b + 0.3
                                for s, _ in quiet_runs(t, db, np.percentile(ref, 90) - END_GAP_DB, 0.15)):
            out.append("end")
        t, db = fine_loudness(src, a - 10, a + 10)
        ref = db[(t >= a + 2) & (t < a + 10)]
        if len(ref) and not any(a - 0.3 <= e <= a + 2.0
                                for _, e in quiet_runs(t, db, np.percentile(ref, 90) - ONSET_GAP_DB, 0.3)):
            out.append("start")
    except (OSError, subprocess.CalledProcessError):
        pass
    if scores is not None:                      # 後段（開頭 30 秒之後）連續 6 秒以上沒伴奏
        run = 0
        for j in range(max(0, int((a + 30 - WIN / 2) // HOP)), min(int((b - WIN / 2) // HOP), len(scores) - 1) + 1):
            run = run + 1 if scores[j, 1] < 0.3 else 0
            if run >= 2:
                out.append("gap")
                break
    if orig_len and b - a > orig_len + 45:
        out.append("long")
    return out


def check_timeline(d, timeline):
    """timeline 裡每首歌重新檢查切點，結果放在 song["check"]；人確認過沒問題的（ok）和勾了「已手動處理」的不標。
    回傳有問題的首數"""
    src = d / timeline["source"] if timeline.get("source") else None
    scores = np.load(d / "scores.npy") if (d / "scores.npy").exists() else None
    n = 0
    for s in timeline.get("songs", []):
        s.pop("check", None)
        if timeline.get("locked") or s.get("ok") or not src or not src.exists():
            continue
        c = check_song(src, scores, float(s["start"]), float(s["end"]), s.get("orig_len"))
        if c:
            s["check"], n = c, n + 1
    return n


def extend_intro(segs, scores, loud, music_min=OUTRO_MUSIC, drop=OUTRO_DROP, max_ext=OUTRO_MAX, trim=2.0, fixed=()):
    """跟 extend_outro 對稱：邊講話邊放前奏的，開頭往前延到前奏開始，不超過上一首的結尾。
    fixed：這幾段的開頭不動（粉絲歌單給的開始時間）"""
    if loud is None:
        return list(segs)
    music = scores[:, 1]
    n = min(len(music), len(loud))
    out = []
    for k, (a, b) in enumerate(segs):
        i0 = max(0, int((a - WIN / 2) // HOP))
        i1 = min(n - 1, int((b - WIN / 2) // HOP))
        if i1 < i0 or k in fixed:
            out.append((a, b))
            continue
        lo = out[-1][1] if out else 0.0
        med = float(np.median(loud[i0:i1 + 1]))
        j, first = i0 - 1, None
        while j >= 0 and a - (j * HOP + WIN / 2) <= max_ext and music[j] >= music_min and loud[j] >= med - drop:
            first, j = j, j - 1
        if first is not None:
            a = min(a, max(lo, round(first * HOP + WIN / 2 - HOP / 2 + trim, 1)))
        out.append((a, b))
    return out


def write_scores_csv(path, scores, loud=None):
    s = score_of(scores, loud)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "time_s", "sing", "music", "speech", "score", "loudness_db"])
        for i, row in enumerate(scores):
            t = i * HOP + WIN / 2
            db = round(float(loud[i]), 1) if loud is not None and i < len(loud) else ""
            w.writerow([fmt_t(t), round(t, 1), *[round(float(v), 3) for v in row], round(float(s[i]), 3), db])


# ───────────────────────── 3. 辨識歌名（選用） ─────────────────────────
def identify(src, segs, engine="auto", model=None, url=None, cookies=None, setlist=None):
    """回傳每段的 {title, artist, lyrics, by} 或 None。
    1. YouTube 影片：先找留言 / 說明欄裡粉絲整理的歌單（免費、最準）；setlist 給了就直接用，不再讀留言
    2. 剩下的：有 Gemini 金鑰就用 Gemini（聽歌詞 → Google 搜尋），否則用 Shazam"""
    results = [None] * len(segs)
    entries = setlist if setlist is not None else (fetch_setlist(url, segs, cookies) if url else [])
    if entries:
        for k, e in enumerate(match_setlist(entries, segs)):
            if e:
                results[k] = {"title": e[1], "artist": e[2], "by": "setlist"}
        print(f"  歌單：{sum(r is not None for r in results)}/{len(segs)} 首")
    todo = [k for k, r in enumerate(results) if r is None]
    if not todo:
        return results
    if engine == "auto":
        engine = "gemini" if gemini_key() else "shazam"
    rest = [segs[k] for k in todo]
    got = identify_gemini(src, rest, model or GEMINI_MODEL) if engine == "gemini" else identify_shazam(src, rest)
    for k, g in zip(todo, got):
        if g:
            results[k] = {**g, "by": engine} if g.get("title") else g
    return results


# ───────── 粉絲歌單（YouTube 留言 / 說明欄） ─────────
TS_RE = re.compile(r"(?<![\d:])(\d{1,2}:\d{2}(?::\d{2})?)(?![\d:])")
# 整行只是聊天 / 流程標記（沒有寫原唱時才判斷，「乾杯 / 長渕剛」這種歌名不會被誤刪）
TALK_LABEL = re.compile(
    r"(op|ed|opening|ending|intro|outro|mc|bgm|talk|free ?talk|opening ?talk|intro ?talk|chat|"
    r"super ?chats?|stream ?starts?|start|end|"
    r"開始.*|.*[/／]\s*start|待機.*|.*トーク|.*告知.*|お知らせ.*|おしらせ.*|.*スパチャ.*|.*スーパーチャット.*|"
    r".*雑談.*|.*挨拶|あいさつ|エンディング|おわり|終わり|休憩|乾杯|.*マシュマロ.*|.*スタート|.*読み|"
    r".*達成.*|.*終了.*|.*感想.*|お手紙.*|.*お知らせ.*|"
    r"set ?list|セトリ|セットリスト|timestamps?|タイムスタンプ|目次)",
    re.I)
SETLIST_HEADER = re.compile(r"set ?list|セトリ|セットリスト|timestamps?|タイムスタンプ|目次", re.I)


def _to_sec(ts):
    return sum(int(p) * 60 ** i for i, p in enumerate(reversed(ts.split(":"))))


def _strip_symbols(text):
    """拿掉開頭的符號、emoji（▶ 🎤 ♪ 【 ～ 等），但保留「『 這類括號給後面處理"""
    i = 0
    while i < len(text) and (text[i].isspace() or unicodedata.category(text[i]) in ("So", "Sm", "Sk", "Pd")
                             or text[i] in "【】[]～~〜:：|｜・#＃*＊>＞→"):
        i += 1
    return text[i:]


def _clean_entry(text):
    """時間戳後面那段 → 歌名（含原唱）。只拿掉開頭的區間結尾時間（～0:07:34】）和曲序（01. / #1 / ① / 1曲目）"""
    text = re.sub(r"(?<![A-Za-z0-9.]):_?[A-Za-z][A-Za-z0-9_\-]*:", "", text)   # YouTube 表情 :virtualhug: :_akubiBii:
    # 「0:04:37～0:07:34」「1:02:03 - 1:05:00」的後半（前面要有～或 -，歌名「3:33」不會被刪）
    text = re.sub(r"^[\s】\]]*[～~〜\-–—]\s*" + TS_RE.pattern + r"[\s】\]]*", "", text)
    text = _strip_symbols(text)
    text = re.sub(r"^[《〈【\[]?(?:\d{1,3}[.．、)）]|#\d{1,3}|[①-⑳]|\d{1,3}曲目|song\s*\d{1,3}\s*[:：.]|m\d{1,3}[.:：]?)"
                  r"[》〉】\]]?\s*", "", text, flags=re.I)              # 曲序：01. / #1 / ① / 1曲目 / 《4曲目》
    return _strip_symbols(text).strip(" \t　]】-–—~～|｜")


def _split_title(text):
    """「歌名 / 原唱」「歌名 // 原唱」「歌名⁄原唱」「歌名 by 原唱」「『歌名』原唱」→ (歌名, 原唱)"""
    m = re.match(r"^[『「](.+?)[』」]\s*[/／⁄]?\s*(.*)$", text)
    if m:
        parts = [m.group(1), m.group(2)]
    else:
        parts = re.split(r"\s*(?://|／|⁄|\s/|/\s|\s-\s|\s–\s|\sby\s)\s*", text, maxsplit=1, flags=re.I)
        if len(parts) == 1 and text.count("/") == 1:                 # 「アイドル/YOASOBI」
            left, right = text.split("/")
            if len(left.strip()) >= 2 and len(right.strip()) >= 2 and not left.strip().isdigit():
                parts = [left, right]

    def drop_romaji(t):                            # 「喜劇 (Comedy)」「星野源 (Gen Hoshino)」→ 拿掉英文翻譯
        t = t.strip()
        if re.search(r"[^\x00-\x7f]", t):
            t = re.sub(r"\s*[(（][\x00-\x7f]+[)）]\s*$", "", t).strip()
        return t

    title = drop_romaji(parts[0])
    artist = drop_romaji(parts[1]) if len(parts) > 1 else None
    return title, artist or None


def parse_setlist(text):
    """粉絲歌單 → [(開始秒數, 歌名, 原唱)]。支援「4:38 歌名 / 原唱」「01.【0:04:37～0:07:34】」再換行寫歌名、
    歌名寫在時間戳上一行等格式"""
    lines = [ln for ln in (text or "").splitlines()]
    ts_lines = [k for k, ln in enumerate(lines) if TS_RE.search(ln)]

    def entry_at(k):
        return _clean_entry(lines[k]) if 0 <= k < len(lines) and lines[k].strip() and not TS_RE.search(lines[k]) \
            and not SETLIST_HEADER.search(lines[k]) else ""

    def neighbor(k, step):
        k += step
        while 0 <= k < len(lines) and not lines[k].strip():
            k += step
        return k

    # 歌名寫在時間戳「上一行」的格式：第一個只有時間的行上面是歌名，而且最後一個只有時間的行下面沒有歌名
    bare = [k for k in ts_lines if not _clean_entry(lines[k][TS_RE.search(lines[k]).end():])
            and not _clean_entry(lines[k][:TS_RE.search(lines[k]).start()])]
    above = bool(bare) and bool(entry_at(neighbor(bare[0], -1))) and not entry_at(neighbor(bare[-1], 1))

    out = []
    for k in ts_lines:
        m = TS_RE.search(lines[k])
        after = lines[k][m.end():]
        while True:                                    # 「5:07 5:07　歌名」「05:07 5:07」：同一個時間戳寫了兩次
            dup = re.match(r"\s*" + TS_RE.pattern, after)
            if not dup or _to_sec(dup.group(1)) != _to_sec(m.group(1)):
                break
            after = after[dup.end():]
        if re.fullmatch(r"\s*[(（].*[)）]\s*", after) or re.fullmatch(r"\s*[(（].*[)）]\s*", lines[k]):
            continue                                   # 「(風になる)」「(1:29:12 風になる)」：聊天中順口哼幾句的備註，不是完整一首
        rest = _clean_entry(after)
        if not rest:                                   # 「恋愛裁判　13:05」：歌名寫在時間前面
            before = _clean_entry(lines[k][:m.start()])
            if before and not RANGE_RE.search(lines[k]) \
                    and not any(TALK_LABEL.fullmatch(x.strip()) for x in re.split(r"[/／]", before)):
                rest = before
        if not rest:
            j = neighbor(k, -1 if above else 1)
            c = entry_at(j)
            if c and not c.startswith("└"):
                rest = c
        if not rest:
            continue
        title, artist = _split_title(rest)
        if not title or (not artist and TALK_LABEL.fullmatch(title.strip())):
            continue
        out.append((_to_sec(m.group(1)), title, artist))
    return out


RANGE_RE = re.compile(r"(?<![\d:])(\d{1,2}:\d{2}(?::\d{2})?)\s*[～~〜\-–—]\s*(\d{1,2}:\d{2}(?::\d{2})?)(?![\d:])")


def setlist_ends(text):
    """歌單裡「0:04:37～0:07:34」這種有寫結束時間的 → {開始秒數: 結束秒數}"""
    ends = {}
    for m in RANGE_RE.finditer(text or ""):
        a, b = _to_sec(m.group(1)), _to_sec(m.group(2))
        if b > a:
            ends[a] = b
    return ends


_len_lock, _itunes_lock, _itunes_last = threading.Lock(), threading.Lock(), [0.0]
VERSION_WORDS = re.compile(r"feat|ver|version|remix|live|edit|cover|acoustic|mix|remaster|tvsize|short|full|instrumental|offvocal|カラオケ")
NOT_ORIGINAL = re.compile(r"tvsize|tvver|short|instrumental|offvocal|カラオケ")


def _itunes_lookup(title, artist):
    """iTunes Search API 查原曲長度（秒）。查不到 None；連線失敗 False（不寫進快取，下次再查）"""
    with _itunes_lock:                                      # 每分鐘約 20 次，排隊
        wait = _itunes_last[0] + ITUNES_GAP - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _itunes_last[0] = time.monotonic()
    term = f"{title} {artist}" if artist else title
    url = "https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"term": term, "entity": "song", "country": "JP", "limit": 15})
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return False
    want, want_artist = norm_title(title), norm_title(artist or "")
    if not want:
        return None
    scored = []
    for r in data.get("results", []):
        name = norm_title(r.get("trackName") or "")
        if not name or not r.get("trackTimeMillis"):
            continue
        # 歌名要一樣；後面多出來的只能是版本字樣（feat.、ver.、remix…），「Lemon」不能配到「Lemonade」
        if name != want and not (name.startswith(want) and VERSION_WORDS.search(name[len(want):])):
            continue
        if NOT_ORIGINAL.search(name) and not NOT_ORIGINAL.search(want):
            continue                                       # TV size、短版、伴奏不是原曲長度
        art = norm_title(r.get("artistName") or "")
        score = 4 * (name == want) + 3 * bool(want_artist and (want_artist in art or art in want_artist))
        scored.append((score, r["trackTimeMillis"] / 1000))
    if not scored:
        return None
    # 分數最高的那幾筆取中位數：同一首歌有很多翻唱版本，不會被某個特別長或特別短的版本帶偏
    top = sorted(t for sc, t in scored if sc == max(x[0] for x in scored))
    return round(top[len(top) // 2])


def song_length(title, artist=None, cache_dir=None):
    """原曲長度（秒），查不到 None。結果存在 song_lengths.json，同一首不會重複查"""
    if not title:
        return None
    key = f"{norm_title(title)}|{norm_title(artist or '')}"
    cache_p = Path(cache_dir) / "song_lengths.json" if cache_dir else None

    def load():
        try:
            return json.loads(cache_p.read_text(encoding="utf-8")) if cache_p else {}
        except Exception:
            return {}

    with _len_lock:
        cache = load()
    if key in cache:
        return cache[key]
    secs = _itunes_lookup(title, artist)
    if secs is False:
        return None
    if cache_p:
        with _len_lock:
            cache = load()
            cache[key] = secs
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            cache_p.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return secs


def apply_setlist(entries, segs, scores, loud=None, threshold=0.5, max_gap=8.0, pad=2.0, extend=15.0,
                  length_of=None):
    """有粉絲歌單時以歌單為準：每首從歌單的時間開始切；有寫結束時間就用，沒寫就切到偵測到的歌結束（不超過下一首）。
    歌單範圍內、但歌單沒寫的偵測段落大多不是歌（BGM、試唱），丟掉；歌單範圍外的（例如歌單只寫了前半場）保留。
    回傳 (段落, 歌名)；歌單外保留下來的段落歌名是 None"""
    entries = sorted(entries, key=lambda e: float(e["t"]))
    total = (len(scores) - 1) * HOP + WIN
    runs = sorted(detect(scores, threshold, 0.0, max_gap, loud=loud))
    talking = _median(scores[:, 2], TALK_SMOOTH) >= 0.5
    out = []
    for k, e in enumerate(entries):
        start = float(e["t"])
        if start >= total:
            continue
        nxt = float(entries[k + 1]["t"]) if k + 1 < len(entries) else total
        lim = min(nxt, start + SETLIST_MAX_SONG, total)
        if e.get("end") and float(e["end"]) > start:
            end = min(float(e["end"]), total)
        else:
            # 在歌單時間附近（60 秒內）開始、至少 30 秒的那段歌唱到哪裡；更後面才開始的是別首，不算
            hit = next(((a, b) for a, b in runs if a <= start + 60 and min(b, lim) - max(a, start) >= 30), None)
            if hit:
                # 間奏、短暫的口白會把一首歌切成好幾段：後面隔不到 QUIET_GAP 秒的也算這首（不超過下一首）
                # 不到 20 秒的小段（間奏裡講話、伴奏變小聲）當橋接過去，但結尾只算 20 秒以上的
                hit_end = bridge = hit[1]
                for a, b in runs:
                    if a > hit[1] and a < lim and a - bridge <= SETLIST_CHAIN_GAP:
                        bridge = max(bridge, min(b, lim))
                        if b - a >= 20:
                            hit_end = max(hit_end, b)
                end = min(widen([(max(hit[0], start), hit_end)], scores, threshold, pad, extend, loud=loud)[0][1], lim)
                trimmed = trim_talk_tail([(start, end)], scores)[0][1]   # 唱完接著聊天的不要
                if trimmed - start >= 60:                                # 剩太短就相信歌單、不切
                    end = trimmed
                # 往後接偶爾會把唱完後的 BGM 也接進來：一般的偵測結果結尾比較早、又有原曲長度時，
                # 選長度比較接近原曲的那個。翻唱加上尾奏常比原曲長 20～40 秒，所以超過原曲 60 秒以上的才不考慮；
                # 沒有原曲長度就相信往後接的結果（偵測結果常在間奏斷開，切成半首）
                det = next((b for a, b in segs if abs(a - start) <= 45 and b > start + 60), None)
                if det is not None and det < end:
                    orig = length_of(e["title"], e.get("artist")) if length_of else None
                    if orig:
                        cands = [x for x in (end, det) if x - start <= orig + 60] or [det]
                        end = min(cands, key=lambda x: abs((x - start) - orig))
            else:                                          # 沒偵測到：用原曲長度，查不到就先給預設長度
                orig = length_of(e["title"], e.get("artist")) if length_of else None
                end = min(start + (orig + pad if orig else SETLIST_GUESS), lim)
                i0, i1 = int((start - WIN / 2) // HOP), int((end - WIN / 2) // HOP)
                talk = talking[max(0, i0):max(i0 + 1, i1)]
                quiet = loud is None or float(np.mean(loud[max(0, i0):max(i0 + 1, i1)])) < loud_threshold(loud)
                if len(talk) and talk.mean() >= 0.8 and quiet:
                    continue                               # 幾乎都在講話、又不大聲：不是歌（「お手紙」「〇〇達成」）
                trimmed = trim_talk_tail([(start, end)], scores)[0][1]
                if trimmed - start < 30:
                    continue
                end = trimmed
        if end > start:
            out.append(((round(start, 1), round(end, 1)),
                        {"title": e["title"], "artist": e.get("artist"), "by": "setlist"}))
    if not out:
        return segs, [None] * len(segs)
    lo, hi = out[0][0][0] - 60, out[-1][0][1] + 60

    def listed(a, b):                     # 跟歌單切出來的歌重疊超過三成
        return any(min(b, y) - max(a, x) > 0.3 * (b - a) for (x, y), _ in out)

    # 歌單範圍外的保留；範圍內、歌單沒寫的短段落大多不是歌，丟掉；但唱了 2 分鐘以上的（粉絲漏寫、括號備註）保留
    extra = [((a, b), None) for a, b in segs if b < lo or a > hi or (b - a >= 120 and not listed(a, b))]
    both = sorted(out + extra, key=lambda x: x[0][0])
    return [x[0] for x in both], [x[1] for x in both]


def match_setlist(entries, segs):
    """每段配一個歌單項目：開始時間在這段開頭前 45 秒到這段前半之間。
    取其中最晚的那個（這段前面的 MC、前奏標記會被歌名本身蓋過）"""
    used, out = set(), []
    for a, b in segs:
        cands = [(t, i) for i, (t, _, _) in enumerate(entries)
                 if i not in used and a - 45 <= t <= a + min(120, (b - a) / 2)]
        if cands:
            i = max(cands)[1]
            used.add(i)
            out.append(entries[i])
        else:
            out.append(None)
    return out


def song_boundaries(loud, n_frames=None):
    """換歌點：音量比前後 1 分鐘的中位數小 6 dB 以上的地方（好幾首連著唱時，歌和歌之間的空檔）"""
    if loud is None:
        return []
    out = []
    for i in range(len(loud)):
        if loud[i] > SILENT_DB and loud[i] <= np.median(loud[max(0, i - 20):i + 21]) - 6:
            out.append(i * HOP + WIN / 2)
    return out


def setlist_plausible(entries, segs, loud=None, bounds=None):
    """歌單的時間是不是這一場的（時間戳錯誤、貼錯場、精華時刻的留言都對不上）。
    用 66 場真實歌唱直播測過：A 場的歌單套到 B 場，誤收 0.7%；真的歌單幾乎都通過。
    1. 至少 60% 的項目，跟偵測到的歌的開頭差 30 秒以內；或
    2. 好幾首連著唱（偵測切不開）時：至少 70% 的項目，跟開頭或換歌點差 15 秒以內"""
    n = len(entries)
    if n < 2:
        return False
    need = 2 if n <= 3 else 3              # 至少要對上幾首（短歌單容許一首小聲沒偵測到）
    starts = np.array([a for a, b in segs]) if segs else np.array([])
    close = sum(bool(len(starts)) and float(np.min(np.abs(starts - e[0]))) <= 30 for e in entries)
    if close >= need and close / n >= 0.6:
        return True
    b = np.array(sorted(list(starts) + list(bounds if bounds is not None else song_boundaries(loud))))
    close2 = sum(bool(len(b)) and float(np.min(np.abs(b - e[0]))) <= 15 for e in entries)
    return close2 >= need and close2 / n >= 0.7


def choose_setlist(texts, segs, loud=None):
    """從說明欄、留言裡挑出真正的歌單：大部分項目都要對得上偵測到的歌（精華時刻的留言對不上），
    對上最多首的那份優先；歌單被拆成好幾則留言（前半 / 後半）時，把其他也對得上的合併進來"""
    def chunks(entries):                    # 時間倒回去 = 另一份清單（例如歌單後面附的精華時刻）
        cur = []
        for e in entries:
            if cur and e[0] < cur[-1][0] - 60:
                yield cur
                cur = []
            cur.append(e)
        if cur:
            yield cur

    scored = []
    bounds = song_boundaries(loud)
    for entries in (c for t in texts for c in chunks(parse_setlist(t))):
        if len(entries) < 2:
            continue
        if not setlist_plausible(entries, segs, bounds=bounds):
            continue                       # 時間對不上這一場：貼錯場、時間戳錯誤、精華時刻
        hit = sum(e is not None for e in match_setlist(entries, segs))
        scored.append((hit, sum(bool(e[2]) for e in entries), entries))
    if not scored:
        return []
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = list(scored[0][2])
    for _, _, entries in scored[1:]:
        lo, hi = min(x[0] for x in best), max(x[0] for x in best)
        if all(e[0] < lo - 60 or e[0] > hi + 60 for e in entries):   # 前半 / 後半：時間不重疊才合併
            best += entries
    return sorted(best)


def fetch_setlist(url, segs, cookies=None, sources=None, loud=None):
    """讀說明欄和留言，挑出對得上這些歌的粉絲歌單。
    先看前 100 則熱門留言（快，大部分歌單都在這裡）；對不上八成的歌，再把全部留言讀一遍，
    連每則留言底下的前幾則回覆也看（有人會把歌單後半貼在自己留言的回覆裡）"""
    def read(max_comments):
        try:
            import yt_dlp
            opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True,
                    "getcomments": True,
                    "extractor_args": {"youtube": {"max_comments": max_comments, "comment_sort": ["top"]}}}
            opts.update(cookie_opts(cookies))
            with youtube_slot(), yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            print(f"  ⚠ 讀不到留言：{e}")
            return None
        comments = info.get("comments") or []
        return [info.get("description") or ""] + [c.get("text") or "" for c in comments], len(comments)

    def covered(entries):
        return sum(e is not None for e in match_setlist(entries, segs))

    def keep_sources(entries, texts):                       # 歌單是從哪幾則留言 / 說明欄來的（給人對照）
        if sources is not None:
            chosen = set(entries)
            sources[:] = [t for t in texts if chosen & set(parse_setlist(t))]
        return entries

    got = read(["100", "all", "20"])
    if got is None:
        return []
    best = choose_setlist(got[0], segs, loud)
    if covered(best) >= 0.8 * len(segs) or got[1] < 100:     # 已經夠了，或留言本來就不到 100 則
        return keep_sources(best, got[0])
    first = got[0]
    print(f"  前 100 則留言的歌單只對上 {covered(best)}/{len(segs)} 首，讀取全部留言…", flush=True)
    got = read(["all", "all", "all", "5"])
    if got is None:
        return keep_sources(best, first)
    full = choose_setlist(got[0], segs, loud)
    print(f"  讀了 {got[1]} 則留言")
    return keep_sources(full, got[0]) if covered(full) >= covered(best) else keep_sources(best, first)


def read_env_file(path=None):
    """讀 .env（KEY=VALUE，# 開頭是註解）。每次都重新讀，改了檔案不用重開網頁介面"""
    env = {}
    try:
        lines = Path(path or ENV_FILE).read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return env
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        env[k.strip()] = v
    return env


def holodex_key():
    return read_env_file().get("HOLODEX_API_KEY") or os.environ.get("HOLODEX_API_KEY")


_hx_lock, _hx_last = threading.Lock(), [0.0]


def clean_holodex_name(name):
    """「星に願いを / Hoshi Ni Negaiwo」「ヒバナ (Hibana)」「レクイエム (feat. 星街すいせい)」→ 原文歌名"""
    title = re.split(r"\s+/\s+", str(name or "").strip())[0].strip()
    # iTunes 的版本字樣「 - New Version -」「(TV Size)」「(Remastered)」
    title = re.sub(r"\s+-\s*[^-]*(?:version|ver\.?|remix|mix|edit|remaster(?:ed)?)[^-]*-?\s*$", "", title, flags=re.I)
    title = re.sub(r"\s*[(（]\s*(?:feat\.?|tv\s*size|remaster)[^)）]*[)）]", "", title, flags=re.I).strip()
    if re.search(r"[^\x00-\x7f]", title):
        title = re.sub(r"\s*[(（][\x00-\x7f]+[)）]\s*$", "", title).strip()
    return title


def fetch_holodex(vid, key):
    """Holodex 上這部影片的歌曲標記和時間戳留言 → {"songs": [{t, end, title, artist}], "comments": [文字]}；查不到 None。
    不用連 YouTube，會員限定影片也不用 cookies"""
    with _hx_lock:
        wait = _hx_last[0] + HOLODEX_GAP - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _hx_last[0] = time.monotonic()
    req = urllib.request.Request(f"{HOLODEX_API}/videos/{vid}?include=songs&c=1", headers={
        "X-APIKEY": key, "Accept": "application/json",
        "User-Agent": "vsong/0.1 (karaoke song splitter)"})        # 沒有 User-Agent 會被 Cloudflare 擋
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  ⚠ Holodex 錯誤（{e.code}）" + ("：請檢查 .env 的 HOLODEX_API_KEY" if e.code in (401, 403) else ""))
        return None
    except Exception as e:
        print(f"  ⚠ 連不上 Holodex：{e}")
        return None
    songs = []
    for s in d.get("songs") or []:
        if not isinstance(s, dict):
            continue
        title = clean_holodex_name(s.get("name"))
        if title and s.get("start") is not None:
            songs.append({"t": float(s["start"]), "title": title, "artist": s.get("original_artist") or None,
                          **({"end": float(s["end"])} if s.get("end") else {})})
    return {"songs": sorted(songs, key=lambda x: x["t"]),
            "comments": [c.get("message") or "" for c in d.get("comments") or [] if isinstance(c, dict)]}


def gemini_key():
    """先看專案資料夾的 .env，再看系統環境變數"""
    env = read_env_file()
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if env.get(k):
            return env[k]
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def parse_json_obj(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _clean(v):
    v = "" if v is None else str(v).strip()
    return None if v.lower() in ("", "null", "none", "unknown", "n/a") else v


def _conf(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def identify_gemini(src, segs, model):
    """翻唱 Shazam 認不出來，改用 Gemini：聽寫一段 GEMINI_CLIP 秒的歌詞 → 用歌詞 Google 搜尋原曲。
    Gemini 憑記憶猜的歌名幾乎都錯，所以一定要搜尋；搜不到就換歌曲另一段再試一次。"""
    try:
        from google import genai
        from google.genai import errors
    except ImportError:
        print("  ⚠ 沒安裝 google-genai，略過歌名辨識（pip install google-genai）")
        return [None] * len(segs)
    try:   # Interactions API 丟的是另一套錯誤類別（狀態碼在 status_code，不是 code）
        from google.genai._interactions import APIError as InteractionsError
    except ImportError:
        InteractionsError = errors.APIError
    api_errors = (errors.APIError, InteractionsError)

    def status(e):
        return getattr(e, "code", None) or getattr(e, "status_code", None)

    def message(e):
        return getattr(e, "message", None) or str(e)

    key = gemini_key()
    if not key:
        print(f"  ⚠ 沒設定 GEMINI_API_KEY（寫在 {ENV_FILE}），略過歌名辨識")
        return [None] * len(segs)

    client = genai.Client(api_key=key)
    state = {"searches": 0, "search_ok": True, "stop": False}
    lock = threading.Lock()         # 好幾首同時辨識，共用的計數和輸出要上鎖

    def ask(inp, search, level):
        kw = {"model": model, "input": inp, "generation_config": {"thinking_level": level}}
        if search:
            kw["tools"] = [{"type": "google_search"}]
        # SDK 預設沒有逾時，Google 那邊沒回應就會永遠等下去，所以每次都要設
        timeout = GEMINI_TIMEOUT * (1.5 if search else 1)
        for attempt in range(4):
            try:
                it = client.interactions.create(**kw, timeout=timeout)
                break
            except api_errors as e:
                # 太頻繁、伺服器忙才多試幾次；逾時、連線問題只再試一次（SDK 自己也會先短暫重試）
                code = status(e)
                # 429 有兩種：每分鐘太頻繁（等一下就好）和額度用完（「check your plan and billing」，等也沒用）
                used_up = code == 429 and "billing" in message(e).lower()
                again = (code in (429, 500, 502, 503, 504) and not used_up) or (code is None and attempt == 0)
                if not again or attempt == 3:
                    raise
                time.sleep(10 * 2 ** attempt)
        with lock:
            for g in (it.usage.grounding_tool_count if it.usage else None) or []:
                state["searches"] += g.count or 0
        return parse_json_obj(it.output_text)

    def listen(a, b, frac):
        dur = min(GEMINI_CLIP, b - a)
        t = a + max(0.0, min((b - a) * frac - dur / 2, b - a - dur))
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "clip.mp3"
            ffmpeg(["-ss", f"{t:.2f}", "-t", f"{dur:.2f}", "-i", str(src),
                    "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(clip)])
            audio = base64.b64encode(clip.read_bytes()).decode("ascii")
        r = ask([{"type": "text", "text": LISTEN_PROMPT},
                 {"type": "audio", "data": audio, "mime_type": "audio/mp3"}], False, "minimal")
        return _clean(r.get("lyrics"))

    def say(text):
        with lock:
            print(f"\r{text}", flush=True)

    def one(k, a, b):
        with _sem("gemini", GEMINI_WORKERS):       # 好幾部同時處理，送給 Gemini 的總數還是不超過 GEMINI_WORKERS
            return one_song(k, a, b)

    def one_song(k, a, b):
        if state["stop"]:
            return None
        found, lyrics = None, None
        try:
            # 第一段認不出來，換一段再試（歌比一段還短時兩段會一樣，就不重送）
            for frac in GEMINI_CLIPS[:1] if b - a <= GEMINI_CLIP else GEMINI_CLIPS:
                heard = listen(a, b, frac)
                lyrics = lyrics or heard
                if not state["search_ok"]:
                    break
                if not heard or len(heard) < 6:
                    continue
                try:
                    s = ask(SEARCH_PROMPT.format(lyrics=heard), True, "low")
                except api_errors as e:
                    if status(e) in (400, 403, 429):
                        # 免費方案不能搜尋、搜尋額度用完：之後只聽寫歌詞，不再浪費時間重試
                        with lock:
                            first, state["search_ok"] = state["search_ok"], False
                        if first:
                            say(f"  ⚠ 無法使用 Google 搜尋（{status(e)}），之後只記下歌詞：{message(e)}")
                    else:                                  # 伺服器一時出錯：這首先跳過，下一首照常搜尋
                        say(f"  ⚠ Google 搜尋暫時失敗（{status(e)}）：{message(e)}")
                    break
                if _clean(s.get("title")) and _conf(s.get("confidence")) >= 0.5:
                    found = {"title": _clean(s["title"]), "artist": _clean(s.get("artist"))}
                    break
        except api_errors as e:
            say(f"  ⚠ Gemini 錯誤（{status(e)}）：{message(e)}")
            if status(e) in (400, 401, 403, 404):          # 金鑰錯、模型名稱錯：後面每首都會一樣失敗
                with lock:
                    first, state["stop"] = not state["stop"], True
                if first:
                    say(f"  ⚠ 停止辨識，請檢查 .env 的 GEMINI_API_KEY 和 --gemini-model（{model}）")
                return None

        rec = dict(found or {})
        if lyrics:
            rec["lyrics"] = lyrics
        short = re.sub(r"\s+", " ", lyrics or "")[:30]
        say(f"  辨識 {k}/{len(segs)}：{found['title'] if found else '（未辨識）' + short}")
        return rec or None

    # 時間都花在等 Google 回應，同時辨識好幾首；結果照原本的順序排
    results = [None] * len(segs)
    with ThreadPoolExecutor(max_workers=max(1, GEMINI_WORKERS)) as pool:
        futs = {pool.submit(keep_tag(one), k, a, b): k for k, (a, b) in enumerate(segs, 1)}
        for fut in as_completed(futs):
            results[futs[fut] - 1] = fut.result()
    print(f"  Gemini 用了 {state['searches']} 次 Google 搜尋")
    return results


def identify_shazam(src, segs):
    try:
        from shazamio import Shazam
    except ImportError:
        print("  ⚠ 沒安裝 shazamio，略過歌名辨識（pip install shazamio）")
        return [None] * len(segs)

    async def run_all():
        sh = Shazam()
        rec = getattr(sh, "recognize", None) or sh.recognize_song
        results = []
        for k, (a, b) in enumerate(segs, 1):
            found = None
            for frac in (0.35, 0.65):
                t = a + (b - a) * frac
                with tempfile.TemporaryDirectory() as td:
                    clip = Path(td) / "clip.mp3"
                    ffmpeg(["-ss", f"{t:.2f}", "-t", "15", "-i", str(src),
                            "-vn", "-ac", "1", "-b:a", "128k", str(clip)])
                    try:
                        r = await rec(str(clip))
                        tr = (r or {}).get("track")
                        if tr:
                            found = {"title": tr.get("title"), "artist": tr.get("subtitle")}
                            break
                    except Exception:
                        pass
            print(f"  辨識 {k}/{len(segs)}：{found['title'] if found else '（未辨識）'}")
            results.append(found)
        return results

    return asyncio.run(run_all())


# ───────────────────────── 4. 切檔 ─────────────────────────
def cut(d, timeline, merge=False):
    src = d / timeline["source"]
    sd = d / "songs"
    if sd.exists():
        for p in sd.iterdir():
            if p.suffix in (".m4a", ".m3u8", ".txt"):
                p.unlink()
    sd.mkdir(exist_ok=True)

    songs = timeline["songs"]
    album = timeline.get("title") or timeline["video_id"]
    lines, files, jobs = ["#EXTM3U"], [], []
    for n, s in enumerate(songs, 1):
        a, b = float(s["start"]), float(s["end"])
        dur = b - a
        title = s.get("title") or f"Song {n} ({fmt_t(a)})"
        fname = song_filename(n, s)
        meta = ["-metadata", f"title={title}",
                "-metadata", f"artist={timeline.get('channel') or ''}",
                "-metadata", f"album={album}",
                "-metadata", f"track={n}/{len(songs)}"]
        if s.get("artist"):
            meta += ["-metadata", f"comment=原唱：{s['artist']}"]
        jobs.append(["-ss", f"{a:.2f}", "-t", f"{dur:.2f}", "-i", str(src), "-vn",
                     "-c:a", "aac", "-b:a", "192k", *meta, str(sd / fname)])
        lines += [f"#EXTINF:{int(dur)},{title}", fname]
        files.append(fname)

    # 每個 ffmpeg 主要只用一個 CPU 核心，同時切好幾首
    done = 0
    with ThreadPoolExecutor(max_workers=CUT_WORKERS) as pool:
        def run(job):
            with _sem("ffmpeg", CUT_WORKERS):      # 好幾部同時切，ffmpeg 總數還是不超過 CUT_WORKERS
                ffmpeg(job)

        for fut in as_completed([pool.submit(run, job) for job in jobs]):
            fut.result()
            done += 1
            print(f"\r  切檔 {done}/{len(jobs)}", end="", flush=True)
    print()
    (sd / "playlist.m3u8").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if merge and files:
        lst = sd / "concat.txt"
        lst.write_text("".join(f"file '{f}'\n" for f in files), encoding="utf-8")
        ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                "-metadata", f"title={album}（歌曲合輯）", str(d / "songs_only.m4a")])
        lst.unlink()


# ───────────────────────── 5. 總歌單（跨場次，重複歌曲全部保留） ─────────────────────────
def build_library(out_root):
    streams = []
    for p in out_root.glob("*/timeline.json"):
        try:
            streams.append((p.parent, json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            continue
    if not streams:
        return
    streams.sort(key=lambda x: (x[1].get("upload_date") or "", x[1].get("video_id", "")))

    entries = []
    for d, tl in streams:
        for n, s in enumerate(tl.get("songs", []), 1):
            f = d / "songs" / song_filename(n, s)
            if not f.exists():
                continue
            entries.append({
                "title": s.get("title"), "artist": s.get("artist"),
                "stream": tl.get("title") or tl["video_id"], "date": tl.get("upload_date"),
                "start": float(s["start"]), "end": float(s["end"]),
                "file": f.relative_to(out_root).as_posix(),
                "url": f"{tl['url']}&t={int(float(s['start']))}s" if tl.get("url") else "",
            })

    groups = {}
    for e in entries:
        if e["title"]:
            groups.setdefault(norm_title(e["title"]), []).append(e)

    def label(e):
        return e["title"] or f"{e['stream']} {fmt_t(e['start'])}"

    # 總歌單
    lines = ["#EXTM3U"]
    for e in entries:
        lines += [f"#EXTINF:{int(e['end'] - e['start'])},{label(e)}", e["file"]]
    (out_root / "all_songs.m3u8").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 同一首歌的各場版本
    vd = out_root / "versions"
    if vd.exists():
        for p in vd.glob("*.m3u8"):
            p.unlink()
    dups = {k: v for k, v in groups.items() if len(v) >= 2}
    if dups:
        vd.mkdir(exist_ok=True)
        for v in dups.values():
            lines = ["#EXTM3U"]
            for e in v:
                lines += [f"#EXTINF:{int(e['end'] - e['start'])},{e['title']}（{fmt_date(e['date'])}）",
                          "../" + e["file"]]
            (vd / (safe_name(v[0]["title"]) + ".m3u8")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 總表（utf-8-sig 讓 Excel 正確顯示中日文）
    with open(out_root / "library.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["歌名", "原唱", "版本數", "直播日期", "直播標題", "開始", "長度", "檔案", "YouTube"])
        for e in entries:
            cnt = len(groups[norm_title(e["title"])]) if e["title"] else 1
            w.writerow([e["title"] or "", e["artist"] or "", cnt, fmt_date(e["date"]), e["stream"],
                        fmt_t(e["start"]), fmt_t(e["end"] - e["start"]), e["file"], e["url"]])

    print(f"\n📚 {out_root.name}：{len(streams)} 場、{len(entries)} 首 → {out_root / 'all_songs.m3u8'}")
    if dups:
        print(f"   {len(dups)} 首歌有多個版本（都保留，另列在 versions/）：")
        for v in sorted(dups.values(), key=len, reverse=True)[:10]:
            print(f"   ・{v[0]['title']} ×{len(v)}")


def build_all(out_root):
    for cd in sorted(p for p in out_root.iterdir() if p.is_dir()) if out_root.exists() else []:
        if any(cd.glob("*/timeline.json")):
            build_library(cd)


def merge_same_titles(segs, names, max_gap=60.0):
    """一首歌被間奏、喊聲切成好幾段時，每段都會被認成同一首。相鄰、隔不到 max_gap 秒、歌名一樣 → 合成一首"""
    out_s, out_n = [], []
    def auto(nm):                       # 粉絲歌單、手動切開的是真的兩首（例如安可又唱一次），不合併
        return (nm or {}).get("by") not in ("setlist", "manual")

    def same_artist(x, y):
        ax, ay = norm_title((x or {}).get("artist") or ""), norm_title((y or {}).get("artist") or "")
        return not (ax and ay) or ax == ay

    for (a, b), nm in zip(segs, names):
        title = norm_title((nm or {}).get("title") or "")
        if out_s and title and title == norm_title((out_n[-1] or {}).get("title") or "") \
                and a - out_s[-1][1] <= max_gap and auto(nm) and auto(out_n[-1]) and same_artist(nm, out_n[-1]):
            out_s[-1] = (out_s[-1][0], b)
            if (nm or {}).get("lyrics") and not out_n[-1].get("lyrics"):
                out_n[-1] = {**out_n[-1], "lyrics": nm["lyrics"]}
            continue
        out_s.append((a, b))
        out_n.append(nm)
    if len(out_s) < len(segs):
        print(f"  合併被切開的同一首歌：{len(segs)} 段 → {len(out_s)} 首")
    return out_s, out_n


def reuse_title(tl_path, a, b, min_iou=0.6, keep=None):
    """之前的 timeline 裡，和這段重疊最多（重疊 / 聯集 ≥ min_iou）、而且有歌名或歌詞的那首。
    用重疊比例而不是起訖時間差，改了 --pad / --extend 讓邊界移動十幾秒也認得出來。
    keep：只沿用這些來源（by）的歌名，例如 ("manual", "setlist")"""
    try:
        old = json.loads(Path(tl_path).read_text(encoding="utf-8")).get("songs", [])
    except Exception:
        return None
    best, best_score = None, 0.0
    for s in old:
        if not (s.get("title") or s.get("lyrics")):
            continue
        if keep is not None and not (s.get("title") and s.get("by") in keep):
            continue
        sa, sb = float(s["start"]), float(s["end"])
        inter = max(0.0, min(b, sb) - max(a, sa))
        union = max(b, sb) - min(a, sa)
        iou = inter / union if union > 0 else 0.0
        cover = inter / (b - a) if b > a else 0.0   # 這段有多少落在舊歌裡（之前合併過的歌，偵測還是會切成好幾段）
        score = iou if iou >= min_iou else (cover * 0.99 if cover >= 0.8 else 0.0)
        if score > best_score:
            best, best_score = s, score
    return {k: best[k] for k in ("title", "artist", "lyrics", "by") if best.get(k)} if best else None


def print_summary(d, timeline):
    print(f"\n✔ {timeline.get('title')}")
    print(f"  共 {len(timeline['songs'])} 首 → {d / 'songs'}")
    for n, s in enumerate(timeline["songs"], 1):
        name = s.get("title") or ""
        link = f"   {timeline['url']}&t={int(s['start'])}s" if timeline.get("url") else ""
        print(f"  {n:02d}  {fmt_t(s['start'])} – {fmt_t(s['end'])}  {name}{link}")


# ───────────────────────── 主流程 ─────────────────────────
def locked_dir(url, out_root):
    """這場在網頁上勾了「已手動處理」就回傳它的資料夾（處理時整場略過，手動切好的時間軸不會被蓋掉）"""
    vid = local_id(Path(url).resolve()) if Path(url).is_file() else youtube_id(url)
    d = find_video_dir(Path(out_root), vid) if vid else None
    try:
        return d if d and json.loads((d / "timeline.json").read_text(encoding="utf-8")).get("locked") else None
    except Exception:
        return None


def refine_edges(segs, names, scores, loud, src, setlist=None, log=print):
    """偵測、套完歌單之後的邊界處理：歌單的開始時間對齊伴奏開始、尾奏延伸、前奏延伸"""
    segs = list(segs)
    # 手動輸入的歌單寫了結束時間的照寫的
    fixed_ends = {round(float(e["t"]), 1) for e in (setlist or {}).get("entries") or []
                  if e.get("end") and (setlist or {}).get("manual")}
    fixed = {i for i, (a, b) in enumerate(segs) if names[i] and round(a, 1) in fixed_ends}
    # 粉絲歌單的開始時間標晚了（前奏被切掉）：往前調到伴奏開始前一點點
    moved = 0
    for i, (a, b) in enumerate(segs):
        if names[i] and loud is not None:
            a2 = setlist_onset(src, scores, loud, a, b, segs[i - 1][1] if i else 0.0)
            if a2 < a:
                segs[i], moved = (a2, b), moved + 1
    if moved:
        log(f"  歌單時間比前奏晚的：{moved} 首往前調到伴奏開始")
    # 結尾切在伴奏結束，不是開始講話的地方
    segs = extend_outro(segs, scores, loud, fixed=fixed)
    # 前奏也一樣；歌單給的開始時間不動
    segs = extend_intro(segs, scores, loud, fixed={i for i, nm in enumerate(names) if nm})
    # 結尾再用細的音量對齊到伴奏衰減完的那一刻
    if loud is not None:
        segs = [(a, b) if i in fixed else
                (a, align_end(src, a, b, segs[i + 1][0] if i + 1 < len(segs) else b + END_FWD))
                for i, (a, b) in enumerate(segs)]
    return segs


def process(url, args):
    print(f"▶ {url}")
    if locked_dir(url, args.out):
        print("  ✋ 已標記「已手動處理」，略過（要重新處理請在網頁歌庫取消勾選）")
        return
    if Path(url).is_file():
        d, meta = import_local(url, Path(args.out), args.channel)
    else:
        d, meta = download(url, Path(args.out), cookies_of(args))

    scores_p, loud_p = d / "scores.npy", d / "loudness.npy"
    loud = None
    if scores_p.exists() and not args.reanalyze:
        print("  使用快取的分析結果")
        scores = np.load(scores_p)
    else:
        wav = d / "analysis.wav"
        print("  轉檔中…")
        ffmpeg(["-i", str(d / meta["source"]), "-vn", "-ac", "1", "-ar", str(SR),
                "-c:a", "pcm_s16le", str(wav)])
        with _gpu_lock:
            scores = analyze(wav, args.batch)
        np.save(scores_p, scores)
        print("  計算音量…")
        loud = loudness(wav, len(scores))           # 用剛轉好的 wav，不用再解碼一次原始檔
        np.save(loud_p, loud)
        wav.unlink(missing_ok=True)
    if loud is None:
        if loud_p.exists() and len(np.load(loud_p)) == len(scores):
            loud = np.load(loud_p)
        else:
            print("  計算音量…")
            loud = loudness(d / meta["source"], len(scores))
            np.save(loud_p, loud)
    write_scores_csv(d / "scores.csv", scores, loud)

    segs = detect(scores, args.threshold, args.min_len, args.max_gap, loud=loud)
    if not segs:
        print("  ⚠ 沒偵測到歌曲，試試調低 --threshold（例如 0.3），或打開 scores.csv 看分數")
        return
    pad, extend = getattr(args, "pad", 2.0), getattr(args, "extend", 15.0)
    segs = widen(segs, scores, args.threshold, pad, extend, loud=loud)
    segs = [(a, b) for a, b in trim_talk_tail(split_long(segs, loud), scores) if b - a >= args.min_len]
    # 粉絲歌單：存過就直接用（--reidentify 會重讀）；找到的歌單存起來給人對照，切歌也以它為準
    try:
        saved = json.loads((d / "timeline.json").read_text(encoding="utf-8")).get("setlist")
    except Exception:
        saved = None
    reidentify = getattr(args, "reidentify", False)
    # 手動輸入的歌單（網頁上貼的）完全相信，--reidentify 也不會蓋掉
    setlist = None if reidentify and not (saved or {}).get("manual") else saved
    if setlist and setlist.get("texts"):
        manual = bool(setlist.get("manual"))
        # 留言來的要先確認對得上偵測到的歌；手動貼的不用比對，直接照解析結果
        entries = parse_setlist("\n".join(setlist["texts"])) if manual else choose_setlist(setlist["texts"], segs, loud)
        if manual and entries and not setlist_plausible(entries, segs, loud):
            print("  ⚠ 手動輸入的歌單跟偵測到的歌對不太上，請確認時間是這一場的（還是會照它切）")
        ends = setlist_ends("\n".join(setlist["texts"]))
        setlist = {"entries": [{"t": t, "title": ti, "artist": ar, **({"end": ends[t]} if t in ends else {})}
                               for t, ti, ar in entries], "texts": setlist["texts"],
                   **({"manual": True} if manual else {}),
                   **({"source": saved["source"]} if (saved or {}).get("source") else {})} if entries else None
    old_tl = {}
    try:
        old_tl = json.loads((d / "timeline.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    holodex_checked = bool(old_tl.get("holodex_checked")) and not reidentify
    hx_key, vid = holodex_key(), youtube_id(meta.get("url") or "")
    if args.identify and hx_key and vid and not holodex_checked and not (setlist or {}).get("manual"):
        hx = fetch_holodex(vid, hx_key)
        if hx is not None:
            holodex_checked = True
            hx_ok = len(hx["songs"]) >= 2 and setlist_plausible(
                [(x["t"], x["title"], x["artist"]) for x in hx["songs"]], segs, loud)
            if len(hx["songs"]) >= 2 and not hx_ok:
                print(f"  ⚠ Holodex 的歌曲標記（{len(hx['songs'])} 首）時間跟這一場對不上，不採用")
            if hx_ok:                                  # 社群整理的歌曲標記：有開始和結束時間
                setlist = {"entries": hx["songs"], "texts": [], "source": "holodex"}
                print(f"  Holodex：歌曲標記 {len(hx['songs'])} 首（有開始和結束時間）")
            elif not setlist and hx["comments"]:
                entries = choose_setlist(hx["comments"], segs, loud)
                if entries:
                    chosen = set(entries)
                    sources = [t for t in hx["comments"] if chosen & set(parse_setlist(t))]
                    ends = setlist_ends("\n".join(sources))
                    setlist = {"entries": [{"t": t, "title": ti, "artist": ar, **({"end": ends[t]} if t in ends else {})}
                                           for t, ti, ar in entries], "texts": sources, "source": "comments"}
                    print(f"  Holodex：時間戳留言裡找到歌單 {len(entries)} 首")
    if args.identify and meta.get("url") and not setlist:
        sources = []
        entries = fetch_setlist(meta["url"], segs, cookies_of(args), sources, loud=loud)
        if entries:
            ends = setlist_ends("\n".join(sources))
            setlist = {"entries": [{"t": t, "title": ti, "artist": ar, **({"end": ends[t]} if t in ends else {})}
                                   for t, ti, ar in entries], "texts": sources}
        elif saved and not saved.get("texts"):
            setlist = saved                                    # 這次讀不到、舊的又沒有原始留言：沿用之前存的
    out_root = Path(args.out)
    use_lengths = not getattr(args, "no_lengths", True)
    length_of = (lambda t, a: song_length(t, a, out_root)) if use_lengths else None
    names = [None] * len(segs)
    if setlist and setlist.get("entries"):
        before = len(segs)
        segs, names = apply_setlist(setlist["entries"], segs, scores, loud, args.threshold, args.max_gap,
                                    pad, extend, length_of=length_of)
        listed = sum(nm is not None for nm in names)
        print(f"  依粉絲歌單切：{listed} 首" + (f"，歌單範圍外另有 {len(segs) - listed} 段" if len(segs) > listed else "")
              + f"（偵測到 {before} 段）")
    segs = refine_edges(segs, names, scores, loud, d / meta["source"], setlist)

    # 重跑（例如改門檻）時，時間差不多的歌沿用之前的歌名，不必再辨識（也省 API 費用）；手動填的歌名永遠優先。
    # Gemini 聽過但認不出來的（只有歌詞）也不再送，避免每次重跑都花錢；可以用網頁上的歌詞搜尋手動查。
    # --reidentify：自動辨識的結果全部重來，只保留手動填的和粉絲歌單的
    keep = ("manual", "setlist") if reidentify else None
    reused = 0
    for i, (a, b) in enumerate(segs):
        prev = reuse_title(d / "timeline.json", a, b, keep=keep)
        if prev and (names[i] is None or prev.get("by") == "manual"):
            names[i] = prev
            reused += prev.get("by") != "setlist"
    todo = [i for i, nm in enumerate(names) if not nm]
    if reused:
        print(f"  沿用之前的辨識結果：{reused} 首")
    if args.identify and todo:
        got = identify(d / meta["source"], [segs[i] for i in todo],
                       getattr(args, "identify_with", "auto"), getattr(args, "gemini_model", None), setlist=[])
        for i, nm in zip(todo, got):
            names[i] = nm
            if reidentify and not (nm or {}).get("title"):
                # 重新辨識失敗（例如 Gemini 額度用完）：保留原本的歌名，不要整個清掉
                prev = reuse_title(d / "timeline.json", *segs[i])
                if prev and prev.get("title"):
                    names[i] = prev
    segs, names = merge_same_titles(segs, names)
    # 短的歌查原曲長度：差很多可能是唱一半被切掉，也可能是 TV size，網頁上會提醒你確認
    if length_of:
        for i, ((a, b), nm) in enumerate(zip(segs, names)):
            if nm and nm.get("title") and b - a < LENGTH_CHECK:
                orig = length_of(nm["title"], nm.get("artist"))
                if orig:
                    names[i] = {**nm, "orig_len": orig}

    timeline = {
        **meta,
        "params": {"threshold": args.threshold, "min_len": args.min_len, "max_gap": args.max_gap,
                   "pad": pad, "extend": extend},
        "songs": [{"start": round(a, 1), "end": round(b, 1),
                   "title": (nm or {}).get("title"), "artist": (nm or {}).get("artist"),
                   **{k: nm[k] for k in ("lyrics", "by", "orig_len") if (nm or {}).get(k)}}
                  for (a, b), nm in zip(segs, names)],
    }
    if setlist or saved:
        timeline["setlist"] = setlist or saved
    if holodex_checked:
        timeline["holodex_checked"] = True          # 查過 Holodex 了，重新處理時不用再查（--reidentify 會重查）
    if locked_dir(meta.get("url") or meta.get("local_path") or "", args.out):
        print("  ✋ 處理途中被標記為「已手動處理」，不覆蓋時間軸")   # 處理到一半才在網頁上勾的
        return
    flagged = check_timeline(d, timeline)
    if flagged:
        print(f"  ⚠ 切點可能不準的：{flagged} 首（網頁上會標出來，可以優先確認）")
    (d / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
    cut(d, timeline, args.merge)
    print_summary(d, timeline)


def youtube_hint(err):
    """常見的 YouTube 錯誤 → 怎麼解決"""
    if "format is not available" in err or "challenge solving failed" in err or "Signature solving failed" in err:
        return ("通常是 YouTube 的 JS 驗證沒有解開：確認裝了 Deno（deno --version），"
                "並更新 yt-dlp：pip install -U \"yt-dlp[default]\"")
    if "confirm you" in err and "bot" in err:
        return "被 YouTube 當成機器人：等一陣子再試、把 --sleep 調長，或用 --cookies 帶登入狀態"
    if "members-only" in err.lower() or "join this channel" in err.lower():
        return "會員限定影片：用 --cookies 指定 cookies.txt（網頁上「會員限定影片」選 cookies.txt 檔案）"
    return None


def cookies_of(args):
    """--cookies（cookies.txt）優先，其次 --cookies-from-browser"""
    return getattr(args, "cookies", None) or getattr(args, "cookies_from_browser", None)


def sleep_range(text):
    """「5-10」→ (5, 10)，「7」→ (7, 7)，「0」→ 不停"""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:[-~～]\s*(\d+(?:\.\d+)?))?\s*", str(text))
    if not m:
        raise argparse.ArgumentTypeError(f"看不懂「{text}」，請寫成 5-10 或 7")
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    return (min(lo, hi), max(lo, hi))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(prog="vsong", description="從直播、錄影或錄音中擷取歌曲段落")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("urls", nargs="*", help="YouTube 網址，或電腦裡的影片 / 音檔 / 資料夾（可多個）")
    ap.add_argument("--out", default="out", help="輸出資料夾（預設 out）")
    ap.add_argument("--threshold", type=float, default=0.5, help="判斷為歌曲的分數門檻（預設 0.5）")
    ap.add_argument("--min-len", type=float, default=60, help="最短歌曲長度秒數（預設 60）")
    ap.add_argument("--max-gap", type=float, default=8, help="小於此秒數的空隙會合併（預設 8）")
    ap.add_argument("--pad", type=float, default=2.0, help="每首歌前後各多留幾秒（預設 2）")
    ap.add_argument("--extend", type=float, default=15.0,
                    help="前奏/尾奏邊講話時，邊界最多往外延伸幾秒（預設 15，0 表示不延伸）")
    ap.add_argument("--identify", action="store_true",
                    help="辨識歌名（.env 有 GEMINI_API_KEY 就用 Gemini 聽歌詞，否則用 Shazam）")
    ap.add_argument("--identify-with", choices=["auto", "gemini", "shazam"], default="auto",
                    help="指定辨識方式（預設 auto）")
    ap.add_argument("--gemini-model", default=GEMINI_MODEL, help=f"Gemini 模型（預設 {GEMINI_MODEL}）")
    ap.add_argument("--reidentify", action="store_true",
                    help="之前自動辨識的歌名全部重來（手動填的和粉絲歌單的會保留），要搭配 --identify")
    ap.add_argument("--merge", action="store_true", help="另外輸出全部歌曲串接的 songs_only.m4a")
    ap.add_argument("--reanalyze", action="store_true", help="忽略快取，重新跑模型")
    ap.add_argument("--recut", metavar="DIR", help="依 timeline.json 重新切檔（手動修改後用），可給影片資料夾或影片 ID")
    ap.add_argument("--cookies", metavar="cookies.txt",
                    help="會員限定影片：從瀏覽器匯出的 cookies.txt（Netscape 格式）。比 --cookies-from-browser 穩")
    ap.add_argument("--cookies-from-browser", metavar="BROWSER", help="會員限定影片：chrome / edge / firefox")
    ap.add_argument("--batch", type=int, default=16, help="GPU 批次大小（預設 16）")
    ap.add_argument("--filter", metavar="關鍵字",
                    help="清單/頻道/資料夾只處理標題（檔名）含這些字的影片，逗號分隔，例如 歌枠,karaoke,歌ってみた")
    ap.add_argument("--limit", type=int, help="每個清單/頻道/資料夾最多處理幾部（依清單順序）")
    ap.add_argument("--channel", metavar="名稱", help=f"電腦裡的檔案要放進哪個頻道資料夾（預設「{LOCAL_CHANNEL}」）")
    ap.add_argument("--force", action="store_true", help="清單中已處理過的影片也重新處理")
    ap.add_argument("--sleep", type=sleep_range, default="5-10", metavar="秒",
                    help="每次連 YouTube 至少隔幾秒，避免被當成機器人（預設 5-10，0 表示不停）")
    ap.add_argument("--jobs", type=int, default=2, metavar="N",
                    help="同時處理幾部影片（預設 2；AI 分析一次一部、連 YouTube 照樣排隊）")
    ap.add_argument("--no-lengths", action="store_true", help="不要查原曲長度（iTunes）")
    ap.add_argument("--library", action="store_true", help="只重建總歌單，不處理影片")
    ap.add_argument("--check", action="store_true",
                    help="不重新處理，只檢查歌庫裡所有切點，標出可能不準的（勾了「已手動處理」的略過）")
    args = ap.parse_args()
    if args.reidentify:            # 只重來不辨識的話，等於把歌名全部清掉
        args.identify = True
    global NET_SLEEP
    NET_SLEEP = args.sleep
    if args.cookies and not Path(args.cookies).is_file():
        sys.exit(f"找不到 cookies 檔案：{args.cookies}")

    if not shutil.which("ffmpeg"):
        sys.exit("找不到 ffmpeg，請先安裝並加入 PATH（winget install Gyan.FFmpeg）")

    out_root = Path(args.out)
    if out_root.exists():
        migrate_old_layout(out_root)

    if args.recut:
        d = Path(args.recut)
        if not (d / "timeline.json").exists():          # 也接受只給影片 ID
            d = find_video_dir(out_root, d.name)
            if not d:
                sys.exit(f"找不到 {args.recut} 的 timeline.json")
        timeline = json.loads((d / "timeline.json").read_text(encoding="utf-8"))
        timeline["songs"].sort(key=lambda s: s["start"])
        check_timeline(d, timeline)                     # 改過的切點重新檢查
        (d / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        cut(d, timeline, args.merge)
        print_summary(d, timeline)
        build_library(d.parent)
        return

    if args.library:
        build_all(out_root)
        return

    if args.check:
        paths = sorted(out_root.glob("*/*/timeline.json"))
        total = 0
        for k, p in enumerate(paths, 1):
            try:
                timeline = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if timeline.get("locked"):
                continue
            n = check_timeline(p.parent, timeline)
            total += n
            p.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{k}/{len(paths)}] {timeline.get('title') or p.parent.name}：{n} 首可能不準" if n
                  else f"[{k}/{len(paths)}] {timeline.get('title') or p.parent.name}：沒問題")
        print(f"\n✔ 檢查完成：共 {total} 首切點可能不準，在網頁歌庫用篩選「有可疑切點」找出來")
        return

    if not args.urls:
        ap.print_help()
        return

    kws = [k.strip() for k in args.filter.split(",") if k.strip()] if args.filter else None
    jobs = expand_urls(args.urls, cookies_of(args), kws, args.limit, cache_dir=out_root)
    todo = []
    for i, (u, vid, from_list) in enumerate(jobs, 1):
        if from_list and vid and not args.force and find_video_dir(out_root, vid):
            print(f"\n[{i}/{len(jobs)}] 已處理過，略過 {u}")
            continue
        todo.append((i, u))

    def run(i, u):
        try:
            process(u, args)
        except Exception as e:
            print(f"  ✖ 失敗：{e}")
            hint = youtube_hint(str(e))
            if hint:
                print(f"  💡 {hint}")

    if args.jobs > 1 and len(todo) > 1:
        # 同時處理好幾部：每行前面標上是哪一部；連 YouTube、AI 分析、ffmpeg、Gemini 都共用排隊和上限
        print(f"\n同時處理 {min(args.jobs, len(todo))} 部（共 {len(todo)} 部）")
        out = TaggedOutput(sys.stdout)
        sys.stdout = out

        def tagged(job):
            out.set_tag(f"[{job[0]}/{len(jobs)}] ")
            try:
                run(*job)
            finally:
                out.end()

        pool = ThreadPoolExecutor(max_workers=args.jobs)
        pending = {pool.submit(tagged, job) for job in todo}
        try:
            while pending:                       # 用逾時輪詢，Ctrl+C 才收得到
                _, pending = wait(pending, timeout=0.5)
        except KeyboardInterrupt:
            sys.stdout = out.raw
            print("\n已中斷。處理到一半的影片下次重新處理就會補上。", flush=True)
            pool.shutdown(wait=False, cancel_futures=True)
            os._exit(130)                        # 不等正在跑的影片做完
        finally:
            sys.stdout = out.raw
        pool.shutdown()
    else:
        for i, u in todo:
            print(f"\n[{i}/{len(jobs)}] ", end="")
            run(i, u)
    build_all(out_root)


if __name__ == "__main__":
    main()
