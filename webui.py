#!/usr/bin/env python3
"""
webui.py — vsong 的本機網頁介面

  python webui.py                 # 開啟 http://127.0.0.1:8765
  python webui.py --port 9000 --out D:/vsong-out --no-browser

只綁定本機（127.0.0.1），不需要額外套件。工作會在背景以子行程執行 vsong.py，
同一時間只跑一個（GPU 一次只做一件事比較快）。
"""

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

import vsong

HERE = Path(__file__).resolve().parent
PAGE = HERE / "webui.html"
VSONG = HERE / "vsong.py"

mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("audio/x-mpegurl", ".m3u8")
mimetypes.add_type("audio/ogg", ".opus")
mimetypes.add_type("audio/ogg", ".ogg")
mimetypes.add_type("audio/flac", ".flac")

# 在子行程開 Windows / macOS 原生的選檔視窗（tkinter 不能在伺服器的執行緒裡開）
PICKER = r"""
import json, sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
exts = sys.argv[2]
if sys.argv[1] == "folder":
    p = filedialog.askdirectory(title="選擇資料夾（會處理裡面所有影音檔）")
    paths = [p] if p else []
elif sys.argv[1] == "cookies":
    p = filedialog.askopenfilename(title="選擇 cookies.txt",
                                   filetypes=[("cookies.txt", "*.txt"), ("所有檔案", "*.*")])
    paths = [p] if p else []
else:
    paths = list(filedialog.askopenfilenames(title="選擇影片或音檔",
                 filetypes=[("影片 / 音檔", exts), ("所有檔案", "*.*")]))
print(json.dumps(paths))
"""


def pick_paths(kind):
    exts = " ".join("*" + e for e in sorted(vsong.MEDIA_EXTS))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    r = subprocess.run([sys.executable, "-c", PICKER, kind, exts], capture_output=True,
                       text=True, encoding="utf-8", creationflags=flags)
    if r.returncode != 0:
        raise ValueError("無法開啟選檔視窗，請直接貼上檔案路徑")
    return [str(Path(p)) for p in json.loads(r.stdout or "[]")]


# ───────────────────────── 背景工作 ─────────────────────────
class Job:
    """一次只跑一個 vsong 子行程，輸出累積在 log 裡給前端輪詢。
    重新切檔這類短工作可以排隊（enqueue）：前一個做完自動接著做，修完一場馬上修下一場也不會卡住"""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.log = ""
        self.label = ""
        self.returncode = None
        self.seq = 0            # 每開一個新工作 +1，前端用來判斷要不要清空畫面
        self.pending = []       # 排隊中的工作 [(args, label)]

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, args, label):
        with self.lock:
            if self.running or self.pending:
                return False
            self._spawn(args, label)
            return True

    def enqueue(self, args, label):
        """有工作在跑就排隊，沒有就馬上開始"""
        with self.lock:
            if self.running or self.pending:
                self.pending.append((args, label))
            else:
                self._spawn(args, label)
            return True

    def _spawn(self, args, label):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        cmd = [sys.executable, "-u", str(VSONG), *args]
        self.seq += 1
        self.label = label
        self.returncode = None
        self.log = "$ vsong " + " ".join(_quote(a) for a in args) + "\n"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     cwd=str(HERE), env=env, creationflags=flags)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _pump(self, proc):
        import codecs
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            text = dec.decode(chunk)
            with self.lock:
                self.log += text
        proc.wait()
        with self.lock:
            self.log += dec.decode(b"", final=True)
            self.returncode = proc.returncode
            self.log += f"\n[結束，代碼 {proc.returncode}]\n"
            if self.pending:                       # 接著做排隊中的下一個
                self._spawn(*self.pending.pop(0))

    def stop(self):
        with self.lock:
            self.pending.clear()
            if self.running:
                self.proc.terminate()
                return True
        return False

    def snapshot(self, since=0):
        with self.lock:
            return {"seq": self.seq, "running": self.running, "label": self.label,
                    "pending": [label for _, label in self.pending],
                    "returncode": self.returncode, "offset": len(self.log), "log": self.log[since:]}


def _quote(a):
    return f'"{a}"' if any(c in a for c in " &") else a


def build_args(opts, out_root):
    """把前端表單轉成 vsong 參數"""
    # 一行一個；檔案路徑可能有空白，所以不能用空白切。Windows「複製路徑」會帶引號，順便拿掉
    urls = [u.strip().strip('"').strip() for u in str(opts.get("urls", "")).splitlines()]
    urls = [u for u in urls if u]
    if not urls:
        raise ValueError("請輸入至少一個網址或檔案路徑")
    bad = [u for u in urls if u.startswith("-")]
    if bad:
        raise ValueError(f"看不懂這一行：{bad[0]}")
    args = [*urls, "--out", str(out_root)]
    floats = ("threshold", "min_len", "max_gap", "pad", "extend")
    for key in (*floats, "batch", "limit"):
        v = opts.get(key)
        if v not in (None, ""):
            args += ["--" + key.replace("_", "-"), str(float(v) if key in floats else int(v))]
    if str(opts.get("filter") or "").strip():
        args += ["--filter", str(opts["filter"]).strip()]
    if str(opts.get("sleep") or "").strip():
        try:
            lo, hi = vsong.sleep_range(opts["sleep"])
        except Exception:
            raise ValueError(f"「影片之間停」看不懂：{opts['sleep']}（請寫成 5-10 或 7）")
        args += ["--sleep", f"{lo:g}-{hi:g}"]
    if str(opts.get("channel") or "").strip():
        args += ["--channel", str(opts["channel"]).strip()]
    if opts.get("cookies") in ("chrome", "edge", "firefox"):
        args += ["--cookies-from-browser", opts["cookies"]]
    elif opts.get("cookies") == "file":
        path = str(opts.get("cookies_file") or "").strip().strip('"').strip()
        if not path or not Path(path).is_file():
            raise ValueError(f"找不到 cookies 檔案：{path or '（沒有選）'}")
        args += ["--cookies", path]
    if str(opts.get("jobs") or "").strip():
        args += ["--jobs", str(max(1, min(6, int(opts["jobs"]))))]
    if opts.get("lengths") is False:
        args.append("--no-lengths")
    for key in ("identify", "reidentify", "merge", "force", "reanalyze"):
        if opts.get(key):
            args.append("--" + key)
    return args


# ───────────────────────── 歌庫 ─────────────────────────
def list_library(out_root):
    """回傳 [{name, streams:[{vid, title, date, url, songs:[...]}]}]，依頻道名稱、直播日期（新到舊）排序"""
    channels = {}
    for p in out_root.glob("*/*/timeline.json"):
        try:
            tl = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        d = p.parent
        songs = []
        for n, s in enumerate(tl.get("songs", []), 1):
            f = d / "songs" / vsong.song_filename(n, s)
            songs.append({
                "n": n, "start": float(s["start"]), "end": float(s["end"]),
                "title": s.get("title"), "artist": s.get("artist"), "lyrics": s.get("lyrics"), "by": s.get("by"), "orig_len": s.get("orig_len"),
                "check": s.get("check") or [], "ok": bool(s.get("ok")),
                "file": f.relative_to(out_root).as_posix() if f.exists() else None,
            })
        channels.setdefault(d.parent.name, []).append({
            "vid": tl.get("video_id") or d.name, "dir": d.relative_to(out_root).as_posix(),
            "title": tl.get("title") or d.name, "date": vsong.fmt_date(tl.get("upload_date")),
            "url": tl.get("url"), "local_path": tl.get("local_path"), "params": tl.get("params") or {},
            "has_scores": (d / "scores.csv").exists(),
            "source": (d / tl["source"]).relative_to(out_root).as_posix()
                      if tl.get("source") and (d / tl["source"]).exists() else None,
            "merged": (d / "songs_only.m4a").exists(),
            "locked": bool(tl.get("locked")),
            "songs": songs,
            "setlist": setlist_view(tl.get("setlist"), songs),
            "setlist_manual": bool((tl.get("setlist") or {}).get("manual")),
            "setlist_source": "manual" if (tl.get("setlist") or {}).get("manual")
                              else (tl.get("setlist") or {}).get("source") or ("comments" if tl.get("setlist") else None),
        })
    return [{"name": name, "streams": sorted(st, key=lambda s: s["date"], reverse=True)}
            for name, st in sorted(channels.items(), key=lambda x: x[0].lower())]


def setlist_view(setlist, songs):
    """存下來的粉絲歌單 → 每一項對上第幾首；每首有沒有在歌單裡（網頁對照用）"""
    if not setlist or not setlist.get("entries"):
        for s in songs:
            s["in_setlist"] = None
        return None
    entries = [(float(e["t"]), e.get("title"), e.get("artist")) for e in setlist["entries"]]
    matched = vsong.match_setlist(entries, [(s["start"], s["end"]) for s in songs])
    which = {}
    for s, e in zip(songs, matched):
        s["in_setlist"] = e is not None
        if e is not None:
            which[e] = s["n"]
    return {"entries": [{"t": t, "title": ti, "artist": ar, "n": which.get((t, ti, ar))} for t, ti, ar in entries],
            "texts": setlist.get("texts") or []}


def preview_setlist(text):
    """貼上的粉絲歌單 → 程式讀出來的 [{t, title, artist, end}]，給人先確認"""
    ends = vsong.setlist_ends(text)
    return [{"t": t, "title": ti, "artist": ar, "end": ends.get(t)} for t, ti, ar in vsong.parse_setlist(text)]


def save_manual_setlist(out_root, rel_dir, text):
    """把手動輸入的粉絲歌單存進 timeline.json；回傳重新處理要用的網址或檔案路徑。text 空的就清掉手動歌單"""
    d = safe_path(out_root, rel_dir)
    p = d / "timeline.json" if d else None
    if not p or not p.exists():
        raise ValueError("找不到這場的 timeline.json")
    tl = json.loads(p.read_text(encoding="utf-8"))
    if tl.get("locked"):
        raise ValueError("這場勾了「已手動處理」，處理時會略過。要套用粉絲歌單請先取消勾選")
    text = str(text or "").strip()
    if text:
        entries = preview_setlist(text)
        if not entries:
            raise ValueError("看不懂這份歌單：每一行要有時間（例如 4:38 歌名 / 原唱）")
        tl["setlist"] = {"entries": [{k: v for k, v in e.items() if v is not None} for e in entries],
                         "texts": [text], "manual": True}
    elif (tl.get("setlist") or {}).get("manual"):
        tl.pop("setlist")
    target = tl.get("url") or tl.get("local_path")
    if not target:
        raise ValueError("這場沒有記錄來源網址或檔案，無法重新切檔")
    if not tl.get("url") and not Path(target).exists():
        raise ValueError(f"找不到原始檔案：{target}（重新切檔需要原本的影片 / 音檔）")
    p.write_text(json.dumps(tl, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def set_locked(out_root, rel_dir, locked):
    """「已手動處理」標記：勾了之後處理影片時整場略過，手動切好的時間軸不會被重新偵測蓋掉"""
    d = safe_path(out_root, rel_dir)
    p = d / "timeline.json" if d else None
    if not p or not p.exists():
        raise ValueError("找不到這場的 timeline.json")
    tl = json.loads(p.read_text(encoding="utf-8"))
    if locked:
        tl["locked"] = True
    else:
        tl.pop("locked", None)
    p.write_text(json.dumps(tl, ensure_ascii=False, indent=2), encoding="utf-8")


def save_timeline(out_root, rel_dir, songs):
    d = safe_path(out_root, rel_dir)
    p = d / "timeline.json" if d else None
    if not p or not p.exists():
        raise ValueError("找不到這場的 timeline.json")
    clean = []
    for s in songs:
        a, b = float(s["start"]), float(s["end"])
        if not 0 <= a < b:
            raise ValueError(f"時間不正確：{s.get('start')} – {s.get('end')}")
        song = {"start": round(a, 2), "end": round(b, 2),         # 精確到 0.01 秒
                "title": (str(s.get("title") or "").strip() or None),
                "artist": (str(s.get("artist") or "").strip() or None)}
        if str(s.get("lyrics") or "").strip():
            song["lyrics"] = str(s["lyrics"]).strip()
        # 歌名來源：網頁上改過歌名的算手動（--reidentify 也會保留），沒改的沿用原本的來源
        if song["title"] and float(s.get("orig_len") or 0) > 0:      # 歌名沒改過才會帶過來
            song["orig_len"] = round(float(s["orig_len"]))
        if s.get("ok"):                     # 人確認過「切點沒問題」：之後不再標可疑
            song["ok"] = True
        by = str(s.get("by") or "").strip()
        if song["title"] and by in ("manual", "setlist", "gemini", "shazam"):
            song["by"] = by
        clean.append(song)
    tl = json.loads(p.read_text(encoding="utf-8"))
    tl["songs"] = sorted(clean, key=lambda s: s["start"])
    tl["locked"] = True        # 在網頁上修過、存檔的場次算「已手動處理」：之後處理影片時略過，不會被重新偵測蓋掉
    p.write_text(json.dumps(tl, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def cut_candidates(src, kind, mark, a, b):
    """候選切點（a～b 秒內）：伴奏結束後 / 開始前的安靜，規則跟程式自動對齊的一樣。
    結尾＝安靜開始後 END_MARGIN 秒，開頭＝安靜結束前 ONSET_MARGIN 秒；len 是安靜多長（越長越像音軌換場）"""
    t, db = vsong.fine_loudness(src, mark - 25, mark + 25)
    if kind == "end":
        ref = db[(t >= mark - 15) & (t < mark - 3)]                # 這首結尾前的音量
        if not len(ref):
            return []
        runs = vsong.quiet_runs(t, db, np.percentile(ref, 90) - vsong.END_GAP_DB, 0.15)
        cands = [(s + vsong.END_MARGIN, e - s) for s, e in runs]
    else:
        ref = db[(t >= mark + 2) & (t < mark + 10)]                # 這首開頭後的音量
        if not len(ref):
            return []
        runs = vsong.quiet_runs(t, db, np.percentile(ref, 90) - vsong.ONSET_GAP_DB, 0.3)
        cands = [(max(0.0, e - vsong.ONSET_MARGIN), e - s) for s, e in runs]
    return [{"t": round(x, 2), "len": round(n, 2)} for x, n in cands if a <= x <= b]


def spectrum(out_root, rel_dir, t, before=8.0, after=8.0, kind=None, mark=None):
    """編輯器看圖用：t 前後的頻譜（0～4 kHz）和音量，每 0.05 秒一格；
    kind（start / end）和 mark（目前的切點）有給的話，一起算候選切點"""
    d = safe_path(out_root, rel_dir)
    p = d / "timeline.json" if d else None
    if not p or not p.exists():
        raise ValueError("找不到這場的 timeline.json")
    tl = json.loads(p.read_text(encoding="utf-8"))
    src = d / (tl.get("source") or "")
    if not tl.get("source") or not src.exists():
        raise ValueError("找不到原始音訊（source.*）")
    a = max(0.0, float(t) - float(before))
    b = float(t) + float(after)
    sr, hop, nfft = 8000, 400, 512
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}",
                          "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    m = len(x) // hop
    x = np.pad(x, (0, nfft))
    win = np.hanning(nfft).astype(np.float32)
    frames = np.stack([x[i * hop:i * hop + nfft] * win for i in range(m)]) if m else np.zeros((0, nfft), np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))[:, :256]                     # 0～4 kHz
    spec = 20 * np.log10(mag.reshape(m, 128, 2).mean(2) + 1e-6)              # 128 格頻率
    img = np.clip((spec + 40) / 70 * 255, 0, 255).astype(np.uint8)          # -40～30 dB → 0～255
    db = 10 * np.log10((x[:m * hop].reshape(m, hop) ** 2).mean(1) + 1e-10) if m else np.zeros(0)
    cands = cut_candidates(src, kind, float(mark), a, b) if kind in ("start", "end") and mark is not None else []
    return {"t0": a, "step": hop / sr, "frames": int(m), "bins": 128, "fmax": sr / 2,
            "spec": base64.b64encode(img.tobytes()).decode(), "db": [round(float(v), 1) for v in db],
            "cands": cands}


def safe_path(out_root, rel):
    """把網址路徑對應到 out 底下的檔案，擋掉 ../ 之類跑出 out 的路徑"""
    root = out_root.resolve()
    p = (root / unquote(rel)).resolve()
    return p if p == root or root in p.parents else None


# ───────────────────────── HTTP ─────────────────────────
def make_handler(out_root, job):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                body = PAGE.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/job":
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                self.send_json(job.snapshot(since))
            elif u.path == "/api/info":
                self.send_json({"gemini": bool(vsong.gemini_key()), "gemini_model": vsong.GEMINI_MODEL})
            elif u.path == "/api/spec":
                q = parse_qs(u.query)
                try:
                    self.send_json(spectrum(out_root, q.get("dir", [""])[0], float(q.get("t", ["0"])[0]),
                                            float(q.get("before", ["8"])[0]), float(q.get("after", ["8"])[0]),
                                            q.get("kind", [None])[0], float(q["m"][0]) if q.get("m") else None))
                except (ValueError, OSError, subprocess.CalledProcessError) as e:
                    self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            elif u.path == "/api/library":
                self.send_json({"out": str(out_root.resolve()), "channels": list_library(out_root)})
            elif u.path.startswith("/files/"):
                self.send_file(u.path[len("/files/"):])
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                body = self.read_json()
                if u.path == "/api/run":
                    ok = job.start(build_args(body, out_root), "處理影片")
                elif u.path == "/api/check":
                    ok = job.enqueue(["--check", "--out", str(out_root)], "檢查切點")
                elif u.path == "/api/rebuild":
                    ok = job.enqueue(["--library", "--out", str(out_root)], "重建總歌單")
                elif u.path == "/api/timeline":
                    d = save_timeline(out_root, body.get("dir", ""), body.get("songs", []))
                    args = ["--recut", str(d), "--out", str(out_root)]
                    ok = job.enqueue(args + (["--merge"] if body.get("merge") else []), "重新切檔：" + d.name)
                elif u.path == "/api/lock":
                    set_locked(out_root, body.get("dir", ""), bool(body.get("locked")))
                    return self.send_json({"ok": True})
                elif u.path == "/api/stop":
                    return self.send_json({"ok": job.stop()})
                elif u.path == "/api/setlist/preview":
                    return self.send_json({"entries": preview_setlist(body.get("text", ""))})
                elif u.path == "/api/setlist":
                    target = save_manual_setlist(out_root, body.get("dir", ""), body.get("text", ""))
                    ok = job.enqueue([target, "--out", str(out_root), "--jobs", "1"], "套用粉絲歌單")
                elif u.path == "/api/pick":
                    kind = body.get("kind") if body.get("kind") in ("folder", "cookies") else "files"
                    return self.send_json({"paths": pick_paths(kind)})
                else:
                    return self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, KeyError, TypeError) as e:
                return self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
            if not ok:
                return self.send_json({"error": "目前有工作在執行，請等它結束"}, HTTPStatus.CONFLICT)
            self.send_json({"ok": True, "seq": job.seq})

        def send_file(self, rel):
            p = safe_path(out_root, rel)
            if not p or not p.is_file():
                return self.send_error(HTTPStatus.NOT_FOUND)
            size = p.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range", "")
            partial = rng.startswith("bytes=")
            if partial:                      # 音訊拖曳進度條需要 Range
                a, _, b = rng[6:].split(",")[0].partition("-")
                try:
                    if a:
                        start, end = int(a), int(b) if b else size - 1
                    else:
                        start = max(0, size - int(b))
                except ValueError:
                    partial = False
                end = min(end, size - 1)
                if partial and start > end:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or p.suffix in (".csv", ".json", ".m3u8"):
                ctype += "; charset=utf-8"
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(p, "rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(1 << 16, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

    return Handler


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="vsong-web", description="vsong 的本機網頁介面")
    ap.add_argument("--out", default="out", help="輸出資料夾（預設 out）")
    ap.add_argument("--port", type=int, default=8765, help="連接埠（預設 8765）")
    ap.add_argument("--no-browser", action="store_true", help="不要自動開啟瀏覽器")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    job = Job()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(out_root, job))
    url = f"http://127.0.0.1:{args.port}"
    print(f"vsong 網頁介面：{url}（輸出資料夾 {out_root}）  按 Ctrl+C 結束")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        job.stop()
        server.server_close()


if __name__ == "__main__":
    main()
