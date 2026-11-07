"""
webui 測試：在背景開一個真的 HTTP 伺服器，不跑模型、不連網路。
"""
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import webui
from test_vsong import needs_ffmpeg, write_stream


@pytest.fixture
def server(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    job = webui.Job()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), webui.make_handler(out, job))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", out, job
    job.stop()
    srv.shutdown()
    srv.server_close()


def get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as r:
        return r.status, dict(r.headers), r.read()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_job(base, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = json.loads(get(base + "/api/job")[2])
        if not s["running"] and s["returncode"] is not None:
            return s
        time.sleep(0.2)
    raise TimeoutError


def test_build_args():
    args = webui.build_args({"urls": " u1 \n u2 ", "threshold": "0.6", "limit": "3", "filter": "歌枠",
                             "cookies": "edge", "identify": True, "force": False, "batch": ""}, "OUT")
    assert args == ["u1", "u2", "--out", "OUT", "--threshold", "0.6", "--limit", "3",
                    "--filter", "歌枠", "--cookies-from-browser", "edge", "--identify"]
    with pytest.raises(ValueError):
        webui.build_args({"urls": "  "}, "OUT")
    # 路徑一行一個，可以有空白，Windows「複製路徑」的引號會拿掉
    args = webui.build_args({"urls": '"D:/錄影/歌回 1.mp4"\r\nhttps://y/x\n', "channel": " Aちゃん "}, "OUT")
    assert args == ["D:/錄影/歌回 1.mp4", "https://y/x", "--out", "OUT", "--channel", "Aちゃん"]
    with pytest.raises(ValueError):
        webui.build_args({"urls": "--out C:/"}, "OUT")
    assert webui.build_args({"urls": "u", "sleep": " 3 - 8 "}, "OUT")[-2:] == ["--sleep", "3-8"]
    with pytest.raises(ValueError):
        webui.build_args({"urls": "u", "sleep": "5; rm"}, "OUT")
    jar = Path(__file__)                                   # 隨便一個存在的檔案
    assert webui.build_args({"urls": "u", "cookies": "file", "cookies_file": f' "{jar}" '}, "OUT")[-2:] == \
        ["--cookies", str(jar)]
    with pytest.raises(ValueError):
        webui.build_args({"urls": "u", "cookies": "file", "cookies_file": "D:/沒有這個檔案.txt"}, "OUT")
    assert webui.build_args({"urls": "u", "jobs": "99", "lengths": False}, "OUT")[-3:] == ["--jobs", "6", "--no-lengths"]
    assert "--no-lengths" not in webui.build_args({"urls": "u", "lengths": True}, "OUT")
    # 不認得的瀏覽器不會被傳進命令列
    assert "--cookies-from-browser" not in webui.build_args({"urls": "u", "cookies": "x; rm"}, "OUT")


def test_page_and_path_traversal(server, tmp_path):
    base, out, _ = server
    (tmp_path / "secret.txt").write_text("x")
    status, headers, body = get(base + "/")
    assert status == 200 and b"vsong" in body
    for bad in ("/files/../secret.txt", "/files/%2e%2e/secret.txt", "/files/..%5Csecret.txt"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(base + bad)
        assert e.value.code == 404


@needs_ffmpeg
def test_library_listing_and_range(server):
    base, out, _ = server
    write_stream(out / "Aちゃん" / "v1", "v1", "20250101", [("ビビデバ", 30, 120), (None, 200, 300)])

    data = json.loads(get(base + "/api/library")[2])
    (ch,) = data["channels"]
    assert ch["name"] == "Aちゃん"
    st = ch["streams"][0]
    assert st["vid"] == "v1" and st["date"] == "2025-01-01" and st["source"] == "Aちゃん/v1/source.m4a"
    assert [s["title"] for s in st["songs"]] == ["ビビデバ", None]
    f = st["songs"][0]["file"]
    assert f == "Aちゃん/v1/songs/01 ビビデバ.m4a"

    url = base + "/files/" + urllib.request.quote(f)
    _, headers, full = get(url)
    assert headers["Content-Type"] == "audio/mp4"
    status, headers, part = get(url, {"Range": "bytes=10-19"})
    assert status == 206 and part == full[10:20]
    assert headers["Content-Range"] == f"bytes 10-19/{len(full)}"


@needs_ffmpeg
def test_save_timeline_and_recut(server):
    base, out, job = server
    d = out / "Ch" / "v1"
    write_stream(d, "v1", "20250101", [(None, 30, 120)])

    status, r = post(base + "/api/timeline", {"dir": "Ch/v1", "songs": [{"start": 10, "end": 5}]})
    assert status == 400

    songs = [{"start": 200, "end": 300, "title": "  後面 ", "by": "manual", "orig_len": 250},
             {"start": 30, "end": 120.04, "title": "", "by": "gemini"}]      # 沒歌名就不記來源
    status, r = post(base + "/api/timeline", {"dir": "Ch/v1", "songs": songs})
    assert status == 200
    assert wait_job(base)["returncode"] == 0

    tl = json.loads((d / "timeline.json").read_text(encoding="utf-8"))
    assert [(s["start"], s["end"], s["title"]) for s in tl["songs"]] == [(30, 120.04, None), (200, 300, "後面")]   # 精確到 0.01 秒
    assert [s.get("by") for s in tl["songs"]] == [None, "manual"]
    assert [s.get("orig_len") for s in tl["songs"]] == [None, 250]           # 存檔不會把原曲長度弄丟
    names = sorted(p.name for p in (d / "songs").glob("*.m4a"))
    assert names == ["01 0h00m30s.m4a", "02 後面.m4a"]
    assert (out / "Ch" / "all_songs.m3u8").exists()

    status, r = post(base + "/api/timeline", {"dir": "../x", "songs": []})
    assert status == 400


def test_setlist_view_marks_matches():
    songs = [{"n": 1, "start": 58.0, "end": 180.0}, {"n": 2, "start": 400.0, "end": 520.0}]
    setlist = {"entries": [{"t": 60, "title": "歌A", "artist": "X"}, {"t": 900, "title": "沒唱", "artist": None}],
               "texts": ["1:00 歌A / X"]}
    view = webui.setlist_view(setlist, songs)
    assert [e["n"] for e in view["entries"]] == [1, None]
    assert [s["in_setlist"] for s in songs] == [True, False]
    songs2 = [{"n": 1, "start": 0.0, "end": 10.0}]
    assert webui.setlist_view(None, songs2) is None and songs2[0]["in_setlist"] is None


@needs_ffmpeg
def test_manual_setlist_save_and_preview(tmp_path):
    out = tmp_path / "out"
    d = out / "Ch" / "v1"
    write_stream(d, "v1", "20250101", [("A", 0, 30)])
    assert webui.preview_setlist("0:04:37～0:07:34 プロポーズ / なとり\n13:13 Overdose") == [
        {"t": 277, "title": "プロポーズ", "artist": "なとり", "end": 454},
        {"t": 793, "title": "Overdose", "artist": None, "end": None}]
    with pytest.raises(ValueError):
        webui.save_manual_setlist(out, "Ch/v1", "沒有時間的文字")
    target = webui.save_manual_setlist(out, "Ch/v1", "4:38 プロポーズ / なとり\n13:13 Overdose")
    assert target == "https://www.youtube.com/watch?v=v1"
    tl = json.loads((d / "timeline.json").read_text(encoding="utf-8"))
    assert tl["setlist"]["manual"] is True and [e["title"] for e in tl["setlist"]["entries"]] == ["プロポーズ", "Overdose"]
    view = webui.list_library(out)[0]["streams"][0]
    assert view["setlist_manual"] is True and len(view["setlist"]["entries"]) == 2
    webui.save_manual_setlist(out, "Ch/v1", "")                 # 清空 = 移除手動歌單
    assert "setlist" not in json.loads((d / "timeline.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        webui.save_manual_setlist(out, "../x", "1:00 A")



@needs_ffmpeg
def test_locked_flag(server):
    base, out, _ = server
    d = out / "Ch" / "v1"
    write_stream(d, "v1", "20250101", [("A", 0, 30)])
    assert json.loads(get(base + "/api/library")[2])["channels"][0]["streams"][0]["locked"] is False
    assert post(base + "/api/lock", {"dir": "Ch/v1", "locked": True}) == (200, {"ok": True})
    tl = json.loads((d / "timeline.json").read_text(encoding="utf-8"))
    assert tl["locked"] is True and tl["songs"]                       # 其他內容不動
    assert json.loads(get(base + "/api/library")[2])["channels"][0]["streams"][0]["locked"] is True
    with pytest.raises(ValueError):                                   # 勾了就不能套用粉絲歌單（會被略過）
        webui.save_manual_setlist(out, "Ch/v1", "1:00 A")
    webui.save_timeline(out, "Ch/v1", [{"start": 0, "end": 20, "title": "A"}])   # 編輯時間軸不受影響、標記留著
    assert json.loads((d / "timeline.json").read_text(encoding="utf-8"))["locked"] is True
    post(base + "/api/lock", {"dir": "Ch/v1", "locked": False})
    assert "locked" not in json.loads((d / "timeline.json").read_text(encoding="utf-8"))
    assert post(base + "/api/lock", {"dir": "../x", "locked": True})[0] == 400


@needs_ffmpeg
def test_spectrum_and_ok_flag(server):
    base, out, _ = server
    d = out / "Ch" / "v1"
    write_stream(d, "v1", "20250101", [("A", 0, 3)])
    s, _, body = get(base + "/api/spec?dir=Ch%2Fv1&t=2&before=2&after=1")
    data = json.loads(body)
    assert s == 200 and data["bins"] == 128 and data["frames"] > 0 and len(data["db"]) == data["frames"]
    assert len(webui.base64.b64decode(data["spec"])) == data["frames"] * data["bins"]
    assert get_status(base + "/api/spec?dir=..%2Fx&t=1") == 400                 # 跑出 out 的路徑擋掉
    s, _, body = get(base + "/api/spec?dir=Ch%2Fv1&t=2&before=2&after=1&kind=end&m=2")
    assert s == 200 and isinstance(json.loads(body)["cands"], list)               # 候選切點
    webui.save_timeline(out, "Ch/v1", [{"start": 0, "end": 2, "title": "A", "ok": True}, {"start": 2, "end": 3, "title": "B"}])
    tl = json.loads((d / "timeline.json").read_text(encoding="utf-8"))
    songs = tl["songs"]
    assert songs[0]["ok"] is True and "ok" not in songs[1]
    assert tl["locked"] is True                                   # 存檔 = 已手動處理
    lib = webui.list_library(out)[0]["streams"][0]["songs"]
    assert lib[0]["ok"] is True and lib[1]["check"] == []


def get_status(url):
    try:
        return get(url)[0]
    except urllib.error.HTTPError as e:
        return e.code


@needs_ffmpeg
def test_saves_queue_instead_of_conflict(server):
    base, out, job = server
    for v in ("v1", "v2"):
        write_stream(out / "Ch" / v, v, "20250101", [("A", 0, 3)])
    s1, _ = post(base + "/api/timeline", {"dir": "Ch/v1", "songs": [{"start": 0, "end": 2.5, "title": "A"}]})
    s2, _ = post(base + "/api/timeline", {"dir": "Ch/v2", "songs": [{"start": 0.25, "end": 2, "title": "A"}]})
    assert (s1, s2) == (200, 200)                                    # 前一個還在切也不會被擋
    end = time.time() + 60
    while (job.running or job.pending) and time.time() < end:
        time.sleep(0.2)
    for v, e in (("v1", 2.5), ("v2", 2)):
        tl = json.loads((out / "Ch" / v / "timeline.json").read_text(encoding="utf-8"))
        assert tl["songs"][0]["end"] == e and tl["locked"] is True
    assert job.returncode == 0


@needs_ffmpeg
def test_export(server, tmp_path):
    base, out, job = server
    write_stream(out / "Ch" / "v1", "v1", "20250101", [("A", 0, 3), ("B", 5, 8)])
    dest = tmp_path / "music"
    for body in ({"dest": "", "songs": [{"dir": "Ch/v1", "start": 0}]},
                 {"dest": "rel", "songs": [{"dir": "Ch/v1", "start": 0}]},
                 {"dest": str(out / "x"), "songs": [{"dir": "Ch/v1", "start": 0}]},
                 {"dest": str(dest), "songs": []},
                 {"dest": str(dest), "songs": [{"dir": "../../etc", "start": 0}]}):
        assert post(base + "/api/export", body)[0] == 400
    status, r = post(base + "/api/export", {"dest": str(dest), "format": "m4a", "songs": [{"dir": "Ch/v1", "start": 5}]})
    assert status == 200
    s = wait_job(base)
    assert s["returncode"] == 0 and s["label"] == "匯出 1 首"
    assert [p.name for p in (dest / "Ch").iterdir()] == ["B(250101) - Ch.m4a"]
    assert "--export" in s["log"]
