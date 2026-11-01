"""
vsong 測試：不需要 GPU、AI 模型或網路。
YouTube 由假的 yt_dlp 模擬，音訊用 ffmpeg 產生的測試音。
"""
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import vsong

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要 ffmpeg")


@pytest.fixture(autouse=True)
def no_real_env_file(tmp_path, monkeypatch):
    """不要讀到使用者自己的 .env（裡面可能有真的金鑰）"""
    monkeypatch.setattr(vsong, "ENV_FILE", tmp_path / "no.env")
    monkeypatch.setattr(vsong, "GEMINI_WORKERS", 1)        # 假 client 是照順序回覆的；並行另外測
    monkeypatch.setattr(vsong, "NET_SLEEP", (0.0, 0.0))     # 測試不用真的等 YouTube 的間隔
    monkeypatch.setattr(vsong, "_net_last", [0.0])


# ───────────────────────── 輔助 ─────────────────────────
def make_audio(path, seconds=720):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"sine=f=440:d={seconds}:sample_rate=8000", "-ac", "1",
                    "-c:a", "aac", "-b:a", "32k", str(path)], check=True)


SONG = [0.5, 0.9, 0.1]        # sing, music, speech
CHAT = [0.0, 0.0, 0.8]
BGM_CHAT = [0.05, 0.5, 0.8]   # 邊放 BGM 邊聊天


def make_scores(spec, n=250):
    """spec: [(start_s, end_s, row)]；每個 frame 中心時間 = i*HOP + WIN/2"""
    t = np.arange(n) * vsong.HOP + vsong.WIN / 2
    sc = np.tile(np.array(CHAT, np.float32), (n, 1))
    for a, b, row in spec:
        sc[(t >= a) & (t < b)] = row
    return sc


def fake_ytdlp(monkeypatch, pages, on_download=None):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def extract_info(self, url, download=False):
            info = pages[url]
            if download and on_download:
                on_download(self.opts, info)
            return info

        def prepare_filename(self, info):
            return self.opts["outtmpl"].replace("%(ext)s", "m4a")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))


def write_stream(d, vid, date, songs, channel="Ch", channel_id=None):
    d.mkdir(parents=True, exist_ok=True)
    make_audio(d / "source.m4a")
    tl = {"video_id": vid, "title": f"歌回 {vid}", "channel": channel, "channel_id": channel_id,
          "url": f"https://www.youtube.com/watch?v={vid}", "upload_date": date, "source": "source.m4a",
          "songs": [{"start": a, "end": b, "title": t, "artist": None} for t, a, b in songs]}
    (d / "timeline.json").write_text(json.dumps(tl, ensure_ascii=False), encoding="utf-8")
    vsong.cut(d, tl)
    return tl


# ───────────────────────── 偵測 ─────────────────────────
def test_detect_finds_songs_and_ignores_chat():
    sc = make_scores([
        (120, 360, SONG),
        (230, 236, [0.1, 0.5, 0.6]),   # 歌曲中短暫掉分，不該被切斷
        (360, 450, BGM_CHAT),          # 邊放 BGM 邊聊天，不該被當成歌
        (450, 700, SONG),
    ])
    segs = vsong.detect(sc)
    assert len(segs) == 2
    (a1, b1), (a2, b2) = segs
    assert abs(a1 - 120) <= 3 and abs(b1 - 360) <= 3
    assert abs(a2 - 450) <= 3 and abs(b2 - 700) <= 3


def test_detect_drops_short_music():
    sc = make_scores([(100, 130, SONG)])          # 30 秒 < 預設最短 60 秒
    assert vsong.detect(sc) == []
    assert len(vsong.detect(sc, min_len=20)) == 1


def test_detect_threshold():
    sc = make_scores([(100, 300, [0.2, 0.6, 0.3])])   # 分數 0.5
    assert len(vsong.detect(sc, threshold=0.4)) == 1
    assert vsong.detect(sc, threshold=0.6) == []


INTRO_TALK = [0.1, 0.8, 0.6]   # 前奏邊講話：分數 0.3，不到門檻但比「門檻 − 0.5」高


def test_widen_extends_over_talky_intro_but_not_bgm_chat():
    sc = make_scores([(105, 120, INTRO_TALK), (120, 360, SONG), (360, 450, BGM_CHAT)])
    segs = vsong.detect(sc)
    (a, b), = segs
    assert abs(a - 120) <= 3                               # 只靠門檻：前奏被切掉
    (a2, b2), = vsong.widen(segs, sc, pad=2)
    assert 101 <= a2 <= 105                                # 延伸到前奏開頭，再多留 2 秒
    assert b2 == pytest.approx(b + 2)                      # 後面是 BGM 聊天，不延伸，只多留 2 秒
    (a3, b3), = vsong.widen(segs, sc, pad=0, extend=0)
    assert (a3, b3) == pytest.approx((a, b))


def test_widen_limits_and_no_overlap():
    # 很長的低分前奏：最多延伸 extend 秒
    sc = make_scores([(0, 200, INTRO_TALK), (200, 400, SONG)], n=200)
    segs = vsong.detect(sc)
    (a, b), = vsong.widen(segs, sc, pad=0, extend=15)
    assert segs[0][0] - 15 <= a < segs[0][0] - 10
    assert b <= (200 - 1) * vsong.HOP + vsong.WIN          # 不超過音檔結尾

    # 兩首中間只隔一點聊天：放寬後不能重疊，切在中間
    sc = make_scores([(60, 180, SONG), (192, 300, SONG)])
    segs = vsong.detect(sc, max_gap=0)
    assert len(segs) == 2
    (a1, b1), (a2, b2) = vsong.widen(segs, sc, pad=8)     # 間隔 12 秒，各多留 8 秒會重疊
    assert b1 == a2 == pytest.approx((segs[0][1] + segs[1][0]) / 2, abs=0.1)
    assert a1 == pytest.approx(segs[0][0] - 8)


KARAOKE = [0.05, 0.5, 0.1]    # 卡拉 OK 伴奏 + 人聲：AST 認不出 singing，分數只有 0.45
TALK_BGM = [0.0, 0.56, 0.56]   # 邊放 BGM 邊聊天
QUIET = [0.0, 0.3, 0.05]       # 安靜的間奏：分數低，但沒人講話


def make_loud(spec, n=250, base=-35.0):
    t = np.arange(n) * vsong.HOP + vsong.WIN / 2
    loud = np.full(n, base, np.float32)
    for a, b, db in spec:
        loud[(t >= a) & (t < b)] = db
    return loud


def test_loudness_finds_karaoke_songs():
    spec = [(120, 360, KARAOKE), (450, 700, KARAOKE)]
    sc = make_scores(spec)
    sc[(sc[:, 2] == CHAT[2])] = TALK_BGM
    loud = make_loud([(120, 360, -20), (450, 700, -21)])
    assert vsong.detect(sc) == []                          # 只看 AST：一首都抓不到
    segs = vsong.detect(sc, loud=loud)
    assert len(segs) == 2
    assert abs(segs[0][0] - 120) <= 3 and abs(segs[1][1] - 700) <= 3


def test_loud_threshold():
    loud = np.concatenate([np.full(100, -35.0), np.full(60, -20.0), np.full(30, -180.0)])  # 聊天、唱歌、數位靜音
    assert -35 < vsong.loud_threshold(loud) < -20          # 靜音不能把分界拉歪
    assert vsong.loud_threshold(np.full(50, -30.0)) == -30  # 整場一樣大聲：音量不起作用


def test_quiet_interlude_merged_but_back_to_back_songs_not():
    # 一首歌中間有 15 秒安靜的間奏（沒人講話）→ 合併成同一首
    sc = make_scores([(60, 150, SONG), (150, 165, QUIET), (165, 300, SONG)])
    segs = vsong.detect(sc)
    assert len(segs) == 1 and abs(segs[0][0] - 60) <= 3 and abs(segs[0][1] - 300) <= 3
    # 間奏時有人講話 → 不合併
    sc = make_scores([(60, 150, SONG), (165, 300, SONG)])
    assert len(vsong.detect(sc, min_len=30)) == 2
    # 兩首完整的歌中間不講話直接接著唱 → 合起來太長，不合併
    sc = make_scores([(60, 300, SONG), (300, 315, QUIET), (315, 560, SONG)])
    assert len(vsong.detect(sc)) == 2


@needs_ffmpeg
def test_loudness_measures_db(tmp_path):
    src = tmp_path / "v.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=60:sample_rate=8000",
                    "-af", "volume='if(lt(t,30),0.05,1)':eval=frame", "-c:a", "aac", "-b:a", "64k", str(src)], check=True)
    n = int((60 - vsong.WIN) // vsong.HOP) + 1
    loud = vsong.loudness(src, n)
    assert len(loud) == n
    assert loud[-1] - loud[0] == pytest.approx(26, abs=3)  # 0.05 倍 ≈ -26 dB


def test_reuse_title_by_overlap(tmp_path):
    tl = tmp_path / "timeline.json"
    tl.write_text(json.dumps({"songs": [
        {"start": 100, "end": 340, "title": "A", "artist": None},
        {"start": 400, "end": 600, "title": None, "lyrics": "ラララ"},
        {"start": 700, "end": 900, "title": None},
    ]}), encoding="utf-8")
    assert vsong.reuse_title(tl, 85, 343) == {"title": "A"}          # 邊界移動十幾秒還認得
    assert vsong.reuse_title(tl, 398, 603) == {"lyrics": "ラララ"}
    assert vsong.reuse_title(tl, 700, 900) is None                    # 什麼都沒有的不沿用
    assert vsong.reuse_title(tl, 300, 420) is None                    # 跨兩首、重疊太少
    assert vsong.reuse_title(tmp_path / "none.json", 0, 10) is None


# ───────────────────────── 命名 ─────────────────────────
def test_song_filename():
    assert vsong.song_filename(1, {"start": 754, "title": None}) == "01 0h12m34s.m4a"
    assert vsong.song_filename(3, {"start": 0, "title": 'A/B:C?"D\'s'}) == "03 A_B_C_D_s.m4a"


def test_norm_title_ignores_case_width_and_spaces():
    assert vsong.norm_title("Stellar Stellar") == vsong.norm_title("stellar  stellar")
    assert vsong.norm_title("ＡＢＣ") == vsong.norm_title("abc")
    assert vsong.norm_title("ビビデバ") != vsong.norm_title("Stellar Stellar")


# ───────────────────────── 播放清單 / 頻道展開 ─────────────────────────
CHANNEL_PAGES = {
    "CH": {"_type": "playlist", "title": "頻道", "entries": [
        {"_type": "url", "ie_key": "YoutubeTab", "url": "TAB_STREAMS"},
        {"_type": "url", "ie_key": "YoutubeTab", "url": "TAB_VIDEOS"}]},
    "TAB_STREAMS": {"_type": "playlist", "title": "直播", "entries": [
        {"id": "v1", "title": "【歌枠】秋の歌回", "live_status": "was_live"},
        {"id": "v2", "title": "【雑談】聊天", "live_status": "was_live"},
        {"id": "v3", "title": "Karaoke night!", "live_status": "was_live"},
        {"id": "v9", "title": "【歌枠】明天", "live_status": "is_upcoming"}]},
    "TAB_VIDEOS": {"_type": "playlist", "title": "影片", "entries": [
        {"id": "v1", "title": "【歌枠】秋の歌回"}]},
    "SINGLE": {"id": "v5", "title": "單一影片"},
}


def test_expand_channel_with_filter_and_dedup(monkeypatch):
    fake_ytdlp(monkeypatch, CHANNEL_PAGES)
    jobs = vsong.expand_urls(["CH", "SINGLE"], keywords=["歌枠", "karaoke"])
    assert [(vid, from_list) for _, vid, from_list in jobs] == [("v1", True), ("v3", True), ("v5", False)]


def test_expand_uses_saved_playlist_when_youtube_blocked(monkeypatch, tmp_path):
    fake_ytdlp(monkeypatch, CHANNEL_PAGES)
    vsong.expand_urls(["CH"], cache_dir=tmp_path)                       # 第一次：存下清單
    fake_ytdlp(monkeypatch, {})                                          # 之後 cookies 過期：讀什麼都失敗
    jobs = vsong.expand_urls(["CH"], keywords=["歌枠", "karaoke"], cache_dir=tmp_path)
    assert [(vid, from_list) for _, vid, from_list in jobs] == [("v1", True), ("v3", True)]
    assert vsong.expand_urls(["OTHER"], cache_dir=tmp_path) == []        # 沒存過的就沒辦法


@needs_ffmpeg
def test_download_uses_saved_audio_without_youtube(monkeypatch, tmp_path):
    out = tmp_path / "out"
    pages = {"https://www.youtube.com/watch?v=aphAfV1Ge8c":
             {"id": "aphAfV1Ge8c", "title": "歌枠", "channel": "Aちゃん", "channel_id": "UCa", "upload_date": "20261020"}}
    fake_ytdlp(monkeypatch, pages, on_download=lambda opts, info: make_audio(opts["outtmpl"].replace("%(ext)s", "m4a"), 5))
    d, meta = vsong.download("https://www.youtube.com/watch?v=aphAfV1Ge8c", out)
    assert (d / "meta.json").exists() and not (d / "timeline.json").exists()
    # 下載完、還沒處理完（沒有 timeline.json）就中斷；之後 YouTube 讀不到也能繼續
    fake_ytdlp(monkeypatch, {})
    d2, meta2 = vsong.download("https://youtu.be/aphAfV1Ge8c", out)
    assert d2 == d and meta2 == meta and meta2["title"] == "歌枠"
    # 舊版下載的（沒有 meta.json）：讀不到 YouTube 就先用已下載的音訊，標記資訊不完整
    (d / "meta.json").unlink()
    d3, meta3 = vsong.download("https://youtu.be/aphAfV1Ge8c", out)
    assert d3 == d and meta3["source"] == "source.m4a" and meta3["meta_partial"] is True
    # 資訊不完整的不算快取：連得上時會重新讀
    (d / "timeline.json").write_text(json.dumps(meta3), encoding="utf-8")
    fake_ytdlp(monkeypatch, pages)
    assert vsong.download("https://www.youtube.com/watch?v=aphAfV1Ge8c", out)[1]["title"] == "歌枠"
    fake_ytdlp(monkeypatch, {})
    with pytest.raises(Exception):
        vsong.download("https://youtu.be/2omY8cYhWpU", out)              # 沒下載過的還是要連 YouTube


def test_expand_limit(monkeypatch):
    fake_ytdlp(monkeypatch, CHANNEL_PAGES)
    jobs = vsong.expand_urls(["CH"], limit=2)
    assert [vid for _, vid, _ in jobs] == ["v1", "v2"]


# ───────────────────────── 切檔 ─────────────────────────
@needs_ffmpeg
def test_cut_outputs(tmp_path):
    d = tmp_path / "stream"
    d.mkdir()
    make_audio(d / "source.m4a")
    tl = {"video_id": "x", "title": "測試歌回", "channel": "Ch", "url": "https://www.youtube.com/watch?v=x",
          "source": "source.m4a",
          "songs": [{"start": 118.5, "end": 358.5, "title": "ビビデバ", "artist": None},
                    {"start": 448.5, "end": 700.5, "title": None, "artist": None}]}
    vsong.cut(d, tl, merge=True)

    names = sorted(p.name for p in (d / "songs").iterdir())
    assert names == ["01 ビビデバ.m4a", "02 0h07m28s.m4a", "playlist.m3u8"]
    assert (d / "songs_only.m4a").exists()
    m3u = (d / "songs" / "playlist.m3u8").read_text(encoding="utf-8")
    assert m3u.startswith("#EXTM3U") and "01 ビビデバ.m4a" in m3u

    # 重新切檔時舊檔要被清掉
    tl["songs"] = tl["songs"][:1]
    vsong.cut(d, tl)
    assert sorted(p.name for p in (d / "songs").iterdir()) == ["01 ビビデバ.m4a", "playlist.m3u8"]


# ───────────────────────── 頻道資料夾 ─────────────────────────
def test_channel_dir_rename_and_collision(tmp_path):
    a = vsong.channel_dir(tmp_path, "UCaaaa111111", "Aちゃん Ch.")
    a2 = vsong.channel_dir(tmp_path, "UCaaaa111111", "Aちゃん（改名後）")   # 改名仍同資料夾
    b = vsong.channel_dir(tmp_path, "UCbbbb222222", "Aちゃん Ch.")         # 不同頻道撞名
    c = vsong.channel_dir(tmp_path, "UCcccc333333", "B/Vtuber?")
    assert a == a2 == tmp_path / "Aちゃん Ch"           # 結尾的「.」在 Windows 不合法，會被去掉
    assert b == tmp_path / "Aちゃん Ch_222222"
    assert c == tmp_path / "B_Vtuber_"


@needs_ffmpeg
def test_download_places_video_in_channel_folder(tmp_path, monkeypatch):
    pages = {"n1": {"id": "n1", "title": "歌回", "channel": "Aちゃん", "channel_id": "UCaaaa111111",
                    "upload_date": "20261020"}}
    fake_ytdlp(monkeypatch, pages,
               on_download=lambda opts, info: make_audio(opts["outtmpl"].replace("%(ext)s", "m4a"), 5))
    d, meta = vsong.download("n1", tmp_path)
    assert d == tmp_path / "Aちゃん" / "n1"
    assert (d / "source.m4a").exists()
    assert meta["channel_id"] == "UCaaaa111111" and meta["source"] == "source.m4a"


@needs_ffmpeg
def test_migrate_old_layout(tmp_path):
    write_stream(tmp_path / "old1", "old1", "20261001", [("ビビデバ", 100, 300)], channel="Aちゃん")
    (tmp_path / "all_songs.m3u8").write_text("old")
    vsong.migrate_old_layout(tmp_path)
    assert (tmp_path / "Aちゃん" / "old1" / "timeline.json").exists()
    assert not (tmp_path / "old1").exists()
    assert not (tmp_path / "all_songs.m3u8").exists()
    # 之後拿到 channel_id 時，要沿用同一個資料夾
    assert vsong.channel_dir(tmp_path, "UCaaaa111111", "Aちゃん") == tmp_path / "Aちゃん"


# ───────────────────────── 總歌單 ─────────────────────────
@needs_ffmpeg
def test_library_keeps_all_versions(tmp_path):
    ch = tmp_path / "Ch"
    write_stream(ch / "a1", "a1", "20261001", [("ビビデバ", 120, 360), (None, 450, 700)])
    write_stream(ch / "c3", "c3", "20261020", [("ビビデバ", 60, 280), ("Stellar Stellar", 300, 520)])
    write_stream(ch / "d4", "d4", "20261025", [("stellar  stellar", 10, 230)])
    vsong.build_all(tmp_path)

    all_songs = (ch / "all_songs.m3u8").read_text(encoding="utf-8").splitlines()
    files = [l for l in all_songs if not l.startswith("#")]
    assert len(files) == 5                                    # 重複的歌全部保留
    assert files[0].startswith("a1/") and files[-1].startswith("d4/")   # 依日期排序

    versions = sorted(p.name for p in (ch / "versions").iterdir())
    assert versions == ["Stellar Stellar.m3u8", "ビビデバ.m3u8"]
    v = (ch / "versions" / "ビビデバ.m3u8").read_text(encoding="utf-8")
    assert "../a1/songs/01 ビビデバ.m4a" in v and "../c3/songs/01 ビビデバ.m4a" in v

    rows = (ch / "library.csv").read_text(encoding="utf-8-sig").splitlines()
    assert rows[0].startswith("歌名,") and len(rows) == 6


# ───────────────────────── 電腦裡的檔案 ─────────────────────────
def make_video(path, seconds=5, acodec="aac"):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5",
                    "-f", "lavfi", "-i", f"sine=f=440:sample_rate=8000", "-t", str(seconds),
                    "-c:v", "mpeg4", "-c:a", acodec, str(path)], check=True)


@needs_ffmpeg
def test_expand_local_folder_without_ytdlp(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)        # 只給本機檔案時不該載入 yt-dlp
    folder = tmp_path / "錄影"
    (folder / "sub").mkdir(parents=True)
    make_audio(folder / "【歌枠】一.m4a", 1)
    make_audio(folder / "sub" / "【歌枠】二.m4a", 2)
    make_audio(folder / "雜談.m4a", 1)
    (folder / "notes.txt").write_text("x")
    single = tmp_path / "單首 有空白.m4a"
    make_audio(single, 1)

    jobs = vsong.expand_urls([str(folder), str(single), str(single)], keywords=["歌枠"])
    assert [(Path(u).name, from_list) for u, _, from_list in jobs] == \
        [("【歌枠】二.m4a", True), ("【歌枠】一.m4a", True), ("單首 有空白.m4a", False)]   # 依完整路徑排序
    assert all(vid.startswith("file-") for _, vid, _ in jobs)


@needs_ffmpeg
@pytest.mark.parametrize("acodec,ext", [("aac", "m4a"), ("pcm_s16le", "m4a")])
def test_import_local_video(tmp_path, acodec, ext):
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    video = src_dir / "歌回 2026.mkv"
    make_video(video, acodec=acodec)
    out = tmp_path / "out"

    d, meta = vsong.import_local(video, out, "Aちゃん")
    assert d.parent == out / "Aちゃん" and d.name == vsong.local_id(video)
    assert meta["source"] == f"source.{ext}" and meta["url"] is None
    assert meta["title"] == "歌回 2026" and meta["local_path"] == str(video.resolve())
    assert len(meta["upload_date"]) == 8
    assert not list(d.glob("*.tmp.*"))

    # 第二次用快取，不重新取音軌
    d2, meta2 = vsong.import_local(video, out, "Aちゃん")
    assert d2 == d and meta2["source"] == meta["source"]


@needs_ffmpeg
def test_process_local_file_end_to_end(tmp_path, monkeypatch):
    audio = tmp_path / "歌回.m4a"
    make_audio(audio, 300)
    monkeypatch.setattr(vsong, "analyze", lambda wav, batch: make_scores([(60, 180, SONG)], n=99))
    out = tmp_path / "out"
    args = types.SimpleNamespace(out=str(out), cookies_from_browser=None, channel=None, reanalyze=False,
                                 threshold=0.5, min_len=60, max_gap=8, identify=False, merge=False, batch=4)
    vsong.process(str(audio), args)
    vsong.build_all(out)

    ch = out / vsong.LOCAL_CHANNEL
    (d,) = [p.parent for p in ch.glob("*/timeline.json")]
    # 偵測到 58.5 秒開始，前面多留 2 秒 → 56.5 秒
    assert [p.name for p in (d / "songs").glob("*.m4a")] == ["01 0h00m56s.m4a"]
    assert "01 0h00m56s.m4a" in (ch / "all_songs.m3u8").read_text(encoding="utf-8")
    rows = (ch / "library.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(rows) == 2 and rows[1].endswith(",")      # 沒有 YouTube 連結


# ───────────────────────── 歌名辨識（Gemini，用假的 client，不連網路） ─────────────────────────
def fake_gemini(monkeypatch, replies):
    """replies：依序回傳的 (文字, 搜尋次數) 或要丟出的例外"""
    from google import genai

    calls = []

    class FakeInteractions:
        def create(self, **kw):
            calls.append(kw)
            r = replies.pop(0)
            if isinstance(r, Exception):
                raise r
            text, searches = r
            usage = types.SimpleNamespace(
                grounding_tool_count=[types.SimpleNamespace(count=searches)] if searches else None)
            return types.SimpleNamespace(output_text=text, usage=usage)

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(genai, "Client", lambda *a, **k: types.SimpleNamespace(interactions=FakeInteractions()))
    monkeypatch.setattr(vsong.time, "sleep", lambda s: None)
    return calls


def gemini_err(code, message="nope"):
    """Interactions API 真正會丟的錯誤類別（不是 google.genai.errors 那套）"""
    import httpx
    from google.genai import _interactions as gx
    if code is None:                                       # 逾時
        return gx.APITimeoutError(request=httpx.Request("POST", "http://x"))
    cls = {400: gx.BadRequestError, 401: gx.AuthenticationError, 429: gx.RateLimitError}.get(code, gx.InternalServerError)
    return cls(message, response=httpx.Response(code, request=httpx.Request("POST", "http://x")), body=None)


def test_parse_json_obj():
    assert vsong.parse_json_obj('```json\n{"title": "怪獣の花唄", "confidence": 0.9}\n```')["title"] == "怪獣の花唄"
    assert vsong.parse_json_obj("不知道") == {} and vsong.parse_json_obj("[1, 2]") == {}


@needs_ffmpeg
def test_gemini_transcribes_then_always_searches(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")
    src = tmp_path / "s.m4a"
    make_audio(src, 400)
    calls = fake_gemini(monkeypatch, [
        # 第 1 首：聽寫歌詞 → 搜尋到
        ('{"lyrics": "思い出すのは君の歌", "singing": true}', 0),
        ('搜尋結果：{"title": "怪獣の花唄", "artist": "Vaundy", "confidence": 0.9}', 2),
        # 第 2 首：第一段搜不到 → 換一段再聽、再搜
        ('{"lyrics": "よくわからない歌詞です", "singing": true}', 0),
        ('{"title": null, "confidence": 0}', 3),
        ('{"lyrics": "夢ならばどれほどよかったでしょう", "singing": true}', 0),
        ('{"title": "Lemon", "artist": "米津玄師", "confidence": 0.9}', 1),
        # 第 3 首：兩段都太短，不搜尋 → 未辨識，但留下歌詞
        ('{"lyrics": "ラララ", "singing": true}', 0),
        ('{"lyrics": "", "singing": false}', 0),
    ])
    got = vsong.identify(src, [(0, 100), (120, 240), (250, 380)])
    assert got[0] == {"title": "怪獣の花唄", "artist": "Vaundy", "lyrics": "思い出すのは君の歌", "by": "gemini"}
    assert got[1]["title"] == "Lemon" and got[1]["lyrics"] == "よくわからない歌詞です"
    assert got[2] == {"lyrics": "ラララ"}
    assert ["tools" in c for c in calls] == [False, True, False, True, False, True, False, False]
    assert calls[0]["model"] == vsong.GEMINI_MODEL
    assert calls[0]["input"][1]["type"] == "audio" and calls[0]["input"][1]["mime_type"] == "audio/mp3"
    assert "夢ならば" in calls[5]["input"]


@needs_ffmpeg
def test_gemini_search_unavailable_and_errors(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")
    src = tmp_path / "s.m4a"
    make_audio(src, 300)
    calls = fake_gemini(monkeypatch, [
        ('{"lyrics": "まだ見ぬ明日へ走れ"}', 0),
        gemini_err(400),                                   # 搜尋不能用（例如免費方案）→ 之後只聽寫歌詞
        ('{"lyrics": "もうひとつの歌詞ですよ"}', 0),
        gemini_err(429), gemini_err(429),                  # 太頻繁 → 重試
        ('{"lyrics": "第三首的歌詞內容"}', 0),
        gemini_err(401),                                   # 金鑰錯誤：直接停止，不再打 API
    ])
    got = vsong.identify(src, [(0, 50), (60, 110), (120, 170), (180, 230), (240, 290)], engine="gemini")
    assert got == [{"lyrics": "まだ見ぬ明日へ走れ"}, {"lyrics": "もうひとつの歌詞ですよ"},
                   {"lyrics": "第三首的歌詞內容"}, None, None]
    assert sum("tools" in c for c in calls) == 1 and len(calls) == 7


def test_gemini_without_key_falls_back_to_shazam(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    used = []
    monkeypatch.setattr(vsong, "identify_shazam", lambda src, segs: used.append("shazam") or [None] * len(segs))
    assert vsong.identify("x", [(0, 1)]) == [None] and used == ["shazam"]


# ───────────────────────── 粉絲歌單 ─────────────────────────
SETLIST_A = """🐢set list🌱
開始/start01:14

01.【00:04:37～0:07:34】
プロポーズ / なとり
└Propose / natori

02.【0:13:12～0:16:25】
Overdose / なとり

[01:37:35] お知らせ
01:40:55 スパチャ読み"""
SETLIST_B = """最高すぎた
4:38 プロポーズ⁄なとりさん
13:13 Overdose
1:38:21～ スパチャ読み&告知"""
SETLIST_C = """00:00      OP
05:40      01.  喜劇 (Comedy) // 星野源 (Gen Hoshino)
1:00:38   09.  orion // 米津玄師 (Kenshi Yonezu)
1:55:17   ED"""
SETLIST_D = """1.2:58 RESISTER / ASCA
5.32:26 Startear / 春奈るな
6.37:43 虹の彼方に /ReoNa
set list
01. 0:05:30 シンクロナイザー"""


def test_parse_setlist_formats():
    assert vsong.parse_setlist(SETLIST_A) == [(277, "プロポーズ", "なとり"), (792, "Overdose", "なとり")]
    assert vsong.parse_setlist(SETLIST_B) == [(278, "プロポーズ", "なとりさん"), (793, "Overdose", None)]
    assert vsong.parse_setlist(SETLIST_C) == [(340, "喜劇", "星野源"), (3638, "orion", "米津玄師")]
    assert vsong.parse_setlist(SETLIST_D) == [(178, "RESISTER", "ASCA"), (1946, "Startear", "春奈るな"),
                                              (2263, "虹の彼方に", "ReoNa"), (330, "シンクロナイザー", None)]
    assert vsong.parse_setlist("") == [] and vsong.parse_setlist(None) == []


def test_match_setlist():
    entries = [(277, "A", None), (792, "B", None), (1281, "C", None)]
    segs = [(270, 460), (700, 990), (1300, 1600), (3000, 3200)]
    # 第 2 段開頭 700：B（792）在前半內；第 3 段開頭 1300：C 在開頭前 19 秒；第 4 段沒有
    assert [e and e[1] for e in vsong.match_setlist(entries, segs)] == ["A", "B", "C", None]


def test_identify_prefers_setlist_then_gemini(monkeypatch):
    monkeypatch.setattr(vsong, "fetch_setlist", lambda url, segs, cookies=None: [(100, "歌單的歌", "原唱")])
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    asked = []
    monkeypatch.setattr(vsong, "identify_gemini", lambda src, segs, model: asked.append(segs) or
                        [{"title": "G", "artist": None, "lyrics": "ラ"} for _ in segs])
    got = vsong.identify("src", [(95, 300), (400, 600)], url="https://www.youtube.com/watch?v=x")
    assert got == [{"title": "歌單的歌", "artist": "原唱", "by": "setlist"},
                   {"title": "G", "artist": None, "lyrics": "ラ", "by": "gemini"}]
    assert asked == [[(400, 600)]]


def test_reuse_title_reidentify_keeps_only_manual_and_setlist(tmp_path):
    tl = tmp_path / "timeline.json"
    tl.write_text(json.dumps({"songs": [
        {"start": 0, "end": 200, "title": "手動", "by": "manual"},
        {"start": 300, "end": 500, "title": "亂猜", "by": "gemini"},
        {"start": 600, "end": 800, "title": "舊的沒有來源"},
        {"start": 900, "end": 1100, "title": "歌單", "by": "setlist"},
    ]}), encoding="utf-8")
    keep = ("manual", "setlist")
    got = [vsong.reuse_title(tl, a, b, keep=keep) for a, b in [(0, 200), (300, 500), (600, 800), (900, 1100)]]
    assert [g and g["title"] for g in got] == ["手動", None, None, "歌單"]
    assert vsong.reuse_title(tl, 300, 500)["title"] == "亂猜"         # 沒有 --reidentify 就沿用


# ───────────────────────── 審查找到的問題 ─────────────────────────
@pytest.mark.parametrize("line,want", [
    ("0:04:37 7 Rings / Ariana Grande", [(277, "7 Rings", "Ariana Grande")]),        # 歌名開頭的數字不是曲序
    ("0:04:37 99 Luftballons / Nena", [(277, "99 Luftballons", "Nena")]),
    ("12:30 3:33 / Junky", [(750, "3:33", "Junky")]),                               # 歌名本身像時間
    ("0:04:37 乾杯 / 長渕剛", [(277, "乾杯", "長渕剛")]),                              # 有原唱就是歌
    ("0:04:37 Start Me Up / Rolling Stones", [(277, "Start Me Up", "Rolling Stones")]),
    ("0:00 待機画面", []), ("3:00 MC", []), ("3:00 Talk", []), ("0:00 歌枠スタート", []),  # 不是歌
    ("3:00 superchats", []), ("1:38:21～ スパチャ読み&告知", []),
    ("0:04:37 アイドル/YOASOBI", [(277, "アイドル", "YOASOBI")]),
    ("0:04:37 『アイドル』/YOASOBI", [(277, "アイドル", "YOASOBI")]),
    ("0:04:37 Lemon by 米津玄師", [(277, "Lemon", "米津玄師")]),
    ("0:04:37 🎤 アイドル / YOASOBI", [(277, "アイドル", "YOASOBI")]),
    ("0:04:37 1曲目 アイドル / YOASOBI", [(277, "アイドル", "YOASOBI")]),
    ("0:04:37 1/3の純情な感情", [(277, "1/3の純情な感情", None)]),
    ("1:02:03 - 1:05:00 Lemon / 米津玄師", [(3723, "Lemon", "米津玄師")]),
])
def test_parse_setlist_tricky_titles(line, want):
    assert vsong.parse_setlist(line) == want


def test_parse_setlist_title_above_timestamp():
    text = "01. プロポーズ / なとり\n0:04:37\n02. Overdose / なとり\n0:13:12\n"
    assert vsong.parse_setlist(text) == [(277, "プロポーズ", "なとり"), (792, "Overdose", "なとり")]


def test_match_setlist_prefers_song_over_talk_label_before_it():
    entries = vsong.parse_setlist("33:05 曲紹介タイム\n33:29 Fly Me To The Moon / Frank Sinatra")
    entries = [(1985, "曲紹介タイム", None)] + entries          # 就算聊天標記沒被濾掉
    assert vsong.match_setlist(entries, [(1987, 2214)])[0][1] == "Fly Me To The Moon"


def test_choose_setlist_ignores_highlight_comments_and_merges_split_setlists():
    segs = [(281, 456), (785, 987), (1277, 1620), (1907, 2187)]
    funny = "1:14 とりあえず張りますね\n10:00 草\n33:47 ﾀｽｶﾙ\n45:00 かわいい\n50:00 www\n1:24:28 合 い す ぎ"
    part1 = "4:38 プロポーズ / なとり\n13:13 Overdose / なとり"
    part2 = "21:22 君の知らない物語 / supercell\n31:53 lulu. / Mrs. GREEN APPLE"
    got = vsong.choose_setlist([funny, part1, part2], segs)
    assert [e[1] for e in got] == ["プロポーズ", "Overdose", "君の知らない物語", "lulu."]
    assert vsong.choose_setlist([funny], segs) == []


def test_quiet_gap_merge_chains():
    H, Q, T = [0.9, 0.9, 0.05], [0.0, 0.4, 0.1], [0.0, 0.1, 0.9]
    mk = lambda spec: np.array([row for row, n in spec for _ in range(n)], np.float32)
    # 一首歌有兩段間奏（沒人講話）→ 三塊合成一首
    segs = vsong.detect(mk([(T, 20), (H, 30), (Q, 5), (H, 25), (Q, 5), (H, 30), (T, 20)]))
    assert len(segs) == 1
    # 201 秒的歌 + 一小段 + 另一首 120 秒的歌：串起來超過 6.5 分鐘 → 不能全部當成一首
    segs = vsong.detect(mk([(T, 20), (H, 67), (Q, 5), (H, 20), (Q, 5), (H, 40), (T, 20)]))
    assert len(segs) == 2


def test_widen_never_cuts_into_the_other_song():
    sc = np.tile(np.float32([[0.9, 0.9, 0.05]]), (100, 1))
    got = vsong.widen([(1.5, 10.5), (19.5, 22.5)], sc, pad=30, extend=40)
    assert got[0][1] <= 19.5 and got[1][0] <= 19.5 and got[1][1] >= 22.5
    assert all(b > a for a, b in got)


@needs_ffmpeg
def test_gemini_search_server_error_does_not_disable_search(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")
    src = tmp_path / "s.m4a"
    make_audio(src, 300)
    calls = fake_gemini(monkeypatch, [
        ('{"lyrics": "思い出すのは君の歌"}', 0),
        *[gemini_err(503)] * 4,                            # 搜尋一直 503：這首跳過
        ('{"lyrics": "夢ならばどれほどよかったでしょう"}', 0),
        ('{"title": "Lemon", "artist": "米津玄師", "confidence": 0.9}', 1),   # 下一首照常搜尋
    ])
    got = vsong.identify(src, [(0, 100), (150, 280)], engine="gemini")
    assert got[0] == {"lyrics": "思い出すのは君の歌"} and got[1]["title"] == "Lemon"
    assert len(calls) == 7


@needs_ffmpeg
def test_gemini_short_song_not_sent_twice(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")
    src = tmp_path / "s.m4a"
    make_audio(src, 60)
    calls = fake_gemini(monkeypatch, [
        ('{"lyrics": "短い歌の歌詞ですよ"}', 0),
        ('{"title": null, "confidence": 0}', 2),
    ])
    assert vsong.identify(src, [(10, 45)], engine="gemini") == [{"lyrics": "短い歌の歌詞ですよ"}]
    assert len(calls) == 2



@needs_ffmpeg
def test_gemini_timeout_skips_song_instead_of_hanging(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")
    src = tmp_path / "s.m4a"
    make_audio(src, 300)
    calls = fake_gemini(monkeypatch, [
        gemini_err(None), gemini_err(None),                # 聽寫一直沒回應：只再試一次，這首跳過
        ('{"lyrics": "夢ならばどれほどよかったでしょう"}', 0),
        ('{"title": "Lemon", "artist": "米津玄師", "confidence": 0.9}', 1),
    ])
    got = vsong.identify(src, [(0, 100), (150, 280)], engine="gemini")
    assert got[0] is None and got[1]["title"] == "Lemon"
    assert len(calls) == 4
    assert all(c["timeout"] > 0 for c in calls)            # 每次請求都有設逾時
    assert calls[3]["timeout"] > calls[2]["timeout"]       # 搜尋等比較久


@needs_ffmpeg
def test_gemini_quota_used_up_is_not_retried(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")
    src = tmp_path / "s.m4a"
    make_audio(src, 300)
    slept = []
    calls = fake_gemini(monkeypatch, [
        ('{"lyrics": "思い出すのは君の歌"}', 0),
        gemini_err(429, "You exceeded your current quota, please check your plan and billing details."),
        ('{"lyrics": "夢ならばどれほどよかったでしょう"}', 0),   # 之後只聽寫，不再搜尋
    ])
    monkeypatch.setattr(vsong.time, "sleep", lambda s: slept.append(s))
    got = vsong.identify(src, [(0, 100), (150, 280)], engine="gemini")
    assert got == [{"lyrics": "思い出すのは君の歌"}, {"lyrics": "夢ならばどれほどよかったでしょう"}]
    assert len(calls) == 3 and slept == []                  # 額度用完：不重試、不等待


@needs_ffmpeg
def test_gemini_runs_songs_in_parallel_and_keeps_order(tmp_path, monkeypatch):
    pytest.importorskip("google.genai")
    import threading
    import time as _time
    from google import genai

    src = tmp_path / "s.m4a"
    make_audio(src, 400)
    lock, state = threading.Lock(), {"n": 0, "active": 0, "peak": 0}

    class FakeInteractions:
        def create(self, **kw):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            _time.sleep(0.2)                                # 模擬等 Google 回應
            with lock:
                state["active"] -= 1
                if "tools" in kw:                           # 搜尋：把聽到的歌詞當成歌名回傳，檢查有沒有配錯首
                    lyr = kw["input"].split("\n")[1]
                    text = json.dumps({"title": lyr, "artist": None, "confidence": 0.9})
                else:
                    state["n"] += 1
                    text = json.dumps({"lyrics": f"第{state['n']}首聽到的歌詞"})
            return types.SimpleNamespace(output_text=text, usage=None)

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(genai, "Client", lambda *a, **k: types.SimpleNamespace(interactions=FakeInteractions()))
    monkeypatch.setattr(vsong, "GEMINI_WORKERS", 4)
    segs = [(i * 60, i * 60 + 50) for i in range(6)]
    got = vsong.identify(src, segs, engine="gemini")
    assert all(g["title"] == g["lyrics"] for g in got)       # 每首的歌名都來自自己的歌詞
    assert len({g["lyrics"] for g in got}) == 6
    assert state["peak"] > 1                                 # 真的有同時處理


@needs_ffmpeg
def test_cut_in_parallel_keeps_playlist_order(tmp_path, monkeypatch):
    monkeypatch.setattr(vsong, "CUT_WORKERS", 4)
    d = tmp_path / "s"
    tl = write_stream(d, "v1", "20250101", [("一", 0, 20), ("二", 30, 50), (None, 60, 80), ("四", 90, 110)])
    m3u = (d / "songs" / "playlist.m3u8").read_text(encoding="utf-8").splitlines()
    assert [l for l in m3u if l.endswith(".m4a")] == ["01 一.m4a", "02 二.m4a", "03 0h01m00s.m4a", "04 四.m4a"]
    assert all((d / "songs" / f).stat().st_size > 0 for f in (l for l in m3u if l.endswith(".m4a")))

@needs_ffmpeg
def test_rerun_keeps_titles_and_only_identifies_new(tmp_path, monkeypatch):
    audio = tmp_path / "歌回.m4a"
    make_audio(audio, 600)
    spec = [(60, 180, SONG)]
    monkeypatch.setattr(vsong, "analyze", lambda wav, batch: make_scores(spec, n=199))
    asked = []
    monkeypatch.setattr(vsong, "identify", lambda src, segs, *a, **k: asked.append(segs) or
                        [{"title": f"曲{len(asked)}", "lyrics": "ラ"} for _ in segs])
    out = tmp_path / "out"
    args = types.SimpleNamespace(out=str(out), cookies_from_browser=None, channel=None, reanalyze=False,
                                 threshold=0.5, min_len=60, max_gap=8, identify=True, merge=False, batch=4)
    vsong.process(str(audio), args)
    (tl_p,) = out.glob("*/*/timeline.json")
    tl = json.loads(tl_p.read_text(encoding="utf-8"))
    assert [(s["title"], s["lyrics"]) for s in tl["songs"]] == [("曲1", "ラ")]

    # 改門檻重跑：多偵測到一首，舊的沿用歌名，只辨識新的那首
    spec.append((300, 420, SONG))
    args.reanalyze = True
    vsong.process(str(audio), args)
    tl = json.loads(tl_p.read_text(encoding="utf-8"))
    assert [s["title"] for s in tl["songs"]] == ["曲1", "曲2"]
    assert len(asked) == 2 and len(asked[1]) == 1

    # 只有歌詞（Gemini 認不出來）的也沿用，不會每次重跑都再送一次
    tl["songs"][1]["title"] = None
    tl_p.write_text(json.dumps(tl, ensure_ascii=False), encoding="utf-8")
    vsong.process(str(audio), args)
    tl = json.loads(tl_p.read_text(encoding="utf-8"))
    assert [(s["title"], s.get("lyrics")) for s in tl["songs"]] == [("曲1", "ラ"), (None, "ラ")]
    assert len(asked) == 2


def test_env_file_key(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('﻿# 註解\n\nexport OTHER=1\nGEMINI_API_KEY = "from-file"\nBROKEN LINE\n', encoding="utf-8")
    assert vsong.read_env_file(env) == {"OTHER": "1", "GEMINI_API_KEY": "from-file"}
    monkeypatch.setattr(vsong, "ENV_FILE", env)
    monkeypatch.setenv("GEMINI_API_KEY", "from-system")
    assert vsong.gemini_key() == "from-file"               # .env 優先
    env.write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    assert vsong.gemini_key() == "from-system"             # .env 空的就用系統環境變數
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert vsong.gemini_key() is None


# ───────────────────────── 影片之間停幾秒 ─────────────────────────
def test_sleep_range():
    assert vsong.sleep_range("5-10") == (5, 10)
    assert vsong.sleep_range(" 7 ") == (7, 7)
    assert vsong.sleep_range("10～3") == (3, 10)
    assert vsong.sleep_range("0") == (0, 0)
    with pytest.raises(Exception):
        vsong.sleep_range("abc")


def test_youtube_wait_spaces_requests(monkeypatch):
    clock, slept = [100.0], []
    monkeypatch.setattr(vsong.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(vsong.time, "sleep", lambda x: (slept.append(x), clock.__setitem__(0, clock[0] + x)))
    monkeypatch.setattr(vsong, "NET_SLEEP", (5.0, 5.0))
    vsong.youtube_wait()                                   # 第一次不用等
    clock[0] += 2
    vsong.youtube_wait()                                   # 上一次才 2 秒前 → 再等 3 秒
    clock[0] += 60
    vsong.youtube_wait()                                   # 隔很久了 → 不用等
    assert slept == [pytest.approx(3.0)]
    monkeypatch.setattr(vsong, "NET_SLEEP", (0.0, 0.0))
    vsong.youtube_wait()
    assert len(slept) == 1


@needs_ffmpeg
def test_download_uses_cache_without_network(tmp_path, monkeypatch):
    d = tmp_path / "Ch" / "abcdefghijk"
    write_stream(d, "abcdefghijk", "20250101", [("A", 0, 30)])
    monkeypatch.setitem(sys.modules, "yt_dlp", None)        # 一連 YouTube 就會出錯
    monkeypatch.setattr(vsong, "youtube_wait", lambda: pytest.fail("不該連 YouTube"))
    got_d, meta = vsong.download("https://www.youtube.com/watch?v=abcdefghijk", tmp_path)
    assert got_d == d and meta["source"] == "source.m4a" and meta["video_id"] == "abcdefghijk"


def test_youtube_id():
    assert vsong.youtube_id("https://www.youtube.com/watch?v=aphAfV1Ge8c&t=3s") == "aphAfV1Ge8c"
    assert vsong.youtube_id("https://youtu.be/aphAfV1Ge8c") == "aphAfV1Ge8c"
    assert vsong.youtube_id("https://www.youtube.com/live/aphAfV1Ge8c?si=x") == "aphAfV1Ge8c"
    assert vsong.youtube_id("D:/錄影/歌回.mp4") is None


def test_tagged_output_prefixes_lines_and_keeps_last_progress():
    import io
    raw = io.StringIO()
    out = vsong.TaggedOutput(raw)
    out.set_tag("[3/9] ")
    out.write("\r  分析中 1/3")
    out.write("\r  分析中 3/3")
    out.write("\n  切檔完成\n\n")
    out.write("沒有換行")
    out.end()
    assert raw.getvalue() == "[3/9] 分析中 3/3\n[3/9] 切檔完成\n[3/9] 沒有換行\n"


def test_runs_several_videos_at_once(tmp_path, monkeypatch):
    import threading
    import time as _time
    jobs = [(f"https://www.youtube.com/watch?v=video{i:06d}", f"video{i:06d}", True) for i in range(6)]
    monkeypatch.setattr(vsong, "expand_urls", lambda *a, **k: jobs)
    lock, state = threading.Lock(), {"active": 0, "peak": 0, "done": []}

    def fake_process(u, args):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        print(f"處理 {u[-6:]}")
        _time.sleep(0.2)
        with lock:
            state["active"] -= 1
            state["done"].append(u)
        if u.endswith("3"):
            raise RuntimeError("壞掉的影片")                # 一部失敗不影響其他部

    monkeypatch.setattr(vsong, "process", fake_process)
    monkeypatch.setattr(sys, "argv", ["vsong", "x", "--out", str(tmp_path / "out"), "--jobs", "3"])
    vsong.main()
    assert len(state["done"]) == 6 and state["peak"] == 3
    assert sys.stdout.__class__.__name__ != "TaggedOutput"    # 跑完要換回原本的輸出


def test_song_length_lookup_and_cache(tmp_path, monkeypatch):
    calls = []
    results = {"results": [
        {"trackName": "Lemonade", "artistName": "X", "trackTimeMillis": 200000},
        {"trackName": "Lemon", "artistName": "Someone", "trackTimeMillis": 250000},
        {"trackName": "Lemon", "artistName": "米津玄師", "trackTimeMillis": 255000},
    ]}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return json.dumps(results).encode("utf-8")

    monkeypatch.setattr(vsong.urllib.request, "urlopen", lambda url, timeout=10: calls.append(url) or Resp())
    monkeypatch.setattr(vsong, "ITUNES_GAP", 0.0)
    assert vsong.song_length("Lemon", "米津玄師", tmp_path) == 255        # 歌名和原唱都對上的優先
    assert vsong.song_length("Lemon", "米津玄師", tmp_path) == 255
    assert len(calls) == 1                                               # 第二次用快取
    results["results"] = [{"trackName": "別的歌", "artistName": "X", "trackTimeMillis": 1000}]
    assert vsong.song_length("Lemon", "誰", tmp_path) is None             # 歌名對不上：寧可查不到
    monkeypatch.setattr(vsong.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("離線")))
    assert vsong.song_length("新歌", None, tmp_path) is None              # 連不上：不寫快取
    assert "新歌|" not in json.loads((tmp_path / "song_lengths.json").read_text(encoding="utf-8"))


def test_apply_setlist_uses_original_length_when_not_detected():
    sc = make_scores([(55, 180, SONG), (500, 620, QUIET)], n=400)       # 500 之後小聲唱、沒偵測到
    entries = [{"t": 62, "title": "A", "artist": None}, {"t": 500, "title": "C", "artist": None}]
    got, names = vsong.apply_setlist(entries, vsong.widen(vsong.detect(sc), sc), sc,
                                     length_of=lambda t, a: 95 if t == "C" else None)
    assert got[1] == (500, 500 + 95 + 2)                              # 原曲 95 秒 + 尾音 2 秒


# ───────────────────────── cookies ─────────────────────────
def test_cookie_opts(tmp_path):
    jar = tmp_path / "youtube cookies.txt"
    jar.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    assert vsong.cookie_opts(None) == {}
    assert vsong.cookie_opts(str(jar)) == {"cookiefile": str(jar)}
    assert vsong.cookie_opts("firefox") == {"cookiesfrombrowser": ("firefox",)}
    args = types.SimpleNamespace(cookies=str(jar), cookies_from_browser="chrome")
    assert vsong.cookies_of(args) == str(jar)                      # cookies.txt 優先


def test_cookies_file_is_passed_to_ytdlp(monkeypatch, tmp_path):
    jar = tmp_path / "c.txt"
    jar.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    seen = []
    fake_ytdlp(monkeypatch, CHANNEL_PAGES)
    real = sys.modules["yt_dlp"].YoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(
        YoutubeDL=lambda opts: seen.append(opts) or real(opts)))
    vsong.expand_urls(["SINGLE"], str(jar))
    assert seen and seen[0]["cookiefile"] == str(jar) and "cookiesfrombrowser" not in seen[0]


def test_missing_cookies_file_stops_early(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["vsong", "x", "--cookies", str(tmp_path / "nope.txt")])
    with pytest.raises(SystemExit) as e:
        vsong.main()
    assert "cookies" in str(e.value)


# ───────────────────────── 讀留言：前 100 則不夠就讀全部 ─────────────────────────
def fake_comments(monkeypatch, top100, everything):
    calls = []

    class YDL:
        def __init__(self, opts):
            self.limit = opts["extractor_args"]["youtube"]["max_comments"][0]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def extract_info(self, url, download=False):
            calls.append(self.limit)
            texts = top100 if self.limit == "100" else everything
            return {"description": "", "comments": [{"text": t} for t in texts]}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=YDL))
    return calls


SEGS4 = [(281, 456), (785, 987), (1277, 1620), (1907, 2187)]
SETLIST4 = "4:38 プロポーズ / なとり\n13:13 Overdose / なとり\n21:22 君の知らない物語 / supercell\n31:53 lulu."


def test_setlist_found_in_top_comments_reads_once(monkeypatch):
    calls = fake_comments(monkeypatch, [SETLIST4] + ["かわいい"] * 120, [])
    assert len(vsong.fetch_setlist("u", SEGS4)) == 4 and calls == ["100"]


def test_setlist_buried_in_comments_reads_all(monkeypatch):
    noise = ["かわいい"] * 100
    calls = fake_comments(monkeypatch, noise, noise + ["草"] * 500 + [SETLIST4])
    got = vsong.fetch_setlist("u", SEGS4)
    assert [e[1] for e in got] == ["プロポーズ", "Overdose", "君の知らない物語", "lulu."]
    assert calls == ["100", "all"]


def test_few_comments_no_second_read(monkeypatch):
    calls = fake_comments(monkeypatch, ["かわいい"] * 30, [SETLIST4])
    assert vsong.fetch_setlist("u", SEGS4) == [] and calls == ["100"]   # 留言不到 100 則，已經全部看過了


# ───────────────────────── 存下粉絲歌單 ─────────────────────────
@needs_ffmpeg
def test_setlist_saved_and_kept_on_rerun(tmp_path, monkeypatch):
    pages = {"n1": {"id": "n1", "title": "歌枠", "channel": "Aちゃん", "channel_id": "UCa", "upload_date": "20261020"}}
    fake_ytdlp(monkeypatch, pages,
               on_download=lambda opts, info: make_audio(opts["outtmpl"].replace("%(ext)s", "m4a"), 600))
    monkeypatch.setattr(vsong, "analyze",
                        lambda wav, batch: make_scores([(60, 180, SONG), (300, 420, SONG), (500, 600, QUIET)], n=199))
    raw = "1:00 歌A / 原唱A\n5:00 歌B\n8:20 沒唱的歌"

    def fetch(url, segs, cookies=None, sources=None, loud=None):
        sources[:] = [raw]
        return [(60, "歌A", "原唱A"), (300, "歌B", None), (500, "沒唱的歌", None)]

    monkeypatch.setattr(vsong, "fetch_setlist", fetch)
    args = types.SimpleNamespace(out=str(tmp_path / "out"), cookies_from_browser=None, channel=None, reanalyze=False,
                                 threshold=0.5, min_len=60, max_gap=8, identify=True, merge=False, batch=4)
    vsong.process("n1", args)
    tl_p = next((tmp_path / "out").glob("*/n1/timeline.json"))
    tl = json.loads(tl_p.read_text(encoding="utf-8"))
    # 以歌單為準：從歌單的時間開始切；偵測沒抓到的「沒唱的歌」也照歌單切出來
    assert [s["title"] for s in tl["songs"]] == ["歌A", "歌B", "沒唱的歌"]
    assert [s["start"] for s in tl["songs"]] == [60, 300, 500]
    assert all(s["by"] == "setlist" for s in tl["songs"])
    assert [e["title"] for e in tl["setlist"]["entries"]] == ["歌A", "歌B", "沒唱的歌"]
    assert tl["setlist"]["texts"] == [raw]

    # 重跑：歌名都沿用、歌單也存過了 → 不再讀留言，歌單保留
    monkeypatch.setattr(vsong, "fetch_setlist", lambda *a, **k: pytest.fail("不該再讀留言"))
    vsong.process("n1", args)
    tl = json.loads(tl_p.read_text(encoding="utf-8"))
    assert len(tl["setlist"]["entries"]) == 3


def test_setlist_ends():
    text = "01.【00:04:37～0:07:34】\n1:02:03 - 1:05:00 Lemon\n12:30 3:33 / Junky\n4:38 プロポーズ"
    assert vsong.setlist_ends(text) == {277: 454, 3723: 3900}


def test_apply_setlist_trusts_fan_timestamps():
    # 偵測：60–180 是歌、245–285 是短 BGM（不在歌單，跟 A 隔一分多鐘，不算 A 的間奏）、900 之後在歌單範圍外
    sc = make_scores([(55, 180, SONG), (245, 285, SONG), (300, 420, SONG), (500, 760, QUIET), (900, 1000, SONG)], n=400)
    segs = vsong.widen(vsong.detect(sc, min_len=30), sc)
    entries = [{"t": 62, "title": "A", "artist": "a"},                 # 只有開始時間
               {"t": 300, "title": "B", "artist": None, "end": 400},  # 有寫結束時間
               {"t": 500, "title": "C", "artist": None}]              # 偵測沒抓到
    got, names = vsong.apply_setlist(entries, segs, sc)
    titles = [n and n["title"] for n in names]
    assert titles == ["A", "B", "C", None]                             # 200–240 的短段落丟掉；900 那段在範圍外保留
    (a1, b1), (a2, b2), (a3, b3), (a4, b4) = got
    assert a1 == 62 and 180 <= b1 <= 200                               # 從歌單時間開始；唱到偵測到的結尾（含一點尾音）
    assert (a2, b2) == (300, 400)                                      # 歌單有結束時間就照用
    assert a3 == 500 and b3 == 500 + vsong.SETLIST_GUESS               # 沒偵測到：先給預設長度
    assert a4 < 900 + 30 and names[3] is None
    assert all(n["by"] == "setlist" for n in names[:3])


def test_apply_setlist_end_never_passes_next_song():
    sc = make_scores([(60, 400, SONG)], n=200)                         # 偵測把兩首連在一起
    entries = [{"t": 60, "title": "A", "artist": None}, {"t": 240, "title": "B", "artist": None}]
    got, names = vsong.apply_setlist(entries, vsong.widen(vsong.detect(sc), sc), sc)
    assert got[0] == (60, 240) and got[1][0] == 240 and [n["title"] for n in names] == ["A", "B"]


def test_merge_same_titles():
    segs = [(0, 100), (120, 200), (200, 260), (400, 500), (520, 600), (900, 1000)]
    names = [{"title": "レオ", "by": "gemini"}, {"title": "ﾚｵ", "by": "gemini", "lyrics": "ラ"},   # 全形半形也算同一首
             {"title": "レオ", "by": "gemini"},
             {"title": "少女レイ"}, None,                                                     # 沒歌名的不合併
             {"title": "少女レイ"}]                                                           # 隔太遠：是又唱了一次
    got_s, got_n = vsong.merge_same_titles(segs, names)
    assert got_s == [(0, 260), (400, 500), (520, 600), (900, 1000)]
    assert got_n[0]["title"] == "レオ" and got_n[0]["lyrics"] == "ラ"


def test_expand_single_video_url_without_network(monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)        # 一連 YouTube 就會出錯
    jobs = vsong.expand_urls(["https://youtu.be/aphAfV1Ge8c?si=x", "https://www.youtube.com/live/2omY8cYhWpU"])
    assert jobs == [("https://www.youtube.com/watch?v=aphAfV1Ge8c", "aphAfV1Ge8c", False),
                    ("https://www.youtube.com/watch?v=2omY8cYhWpU", "2omY8cYhWpU", False)]


@needs_ffmpeg
def test_reidentify_keeps_old_title_when_it_fails(tmp_path, monkeypatch):
    audio = tmp_path / "歌回.m4a"
    make_audio(audio, 400)
    monkeypatch.setattr(vsong, "analyze", lambda wav, batch: make_scores([(60, 180, SONG), (240, 360, SONG)], n=132))
    answers = [[{"title": "舊A", "by": "gemini"}, {"title": "舊B", "by": "gemini"}]]
    monkeypatch.setattr(vsong, "identify", lambda src, segs, *a, **k: answers.pop(0) if answers else
                        [{"title": "新A"}, None])            # 重新辨識：第 1 首成功、第 2 首失敗
    args = types.SimpleNamespace(out=str(tmp_path / "out"), cookies_from_browser=None, channel=None, reanalyze=False,
                                 threshold=0.5, min_len=60, max_gap=8, identify=True, merge=False, batch=4)
    vsong.process(str(audio), args)
    args.reidentify = True
    vsong.process(str(audio), args)
    tl = json.loads(next((tmp_path / "out").glob("*/*/timeline.json")).read_text(encoding="utf-8"))
    assert [s["title"] for s in tl["songs"]] == ["新A", "舊B"]   # 失敗的那首保留原本的歌名


# ───────────────────────── 審查找到的問題（並行、網路、原曲長度） ─────────────────────────
def test_youtube_slot_one_request_at_a_time(monkeypatch):
    import threading
    import time as _time
    monkeypatch.setattr(vsong, "NET_SLEEP", (0.05, 0.05))
    events, lock = [], threading.Lock()

    def request(name):
        with vsong.youtube_slot():
            with lock:
                events.append(("start", name, _time.monotonic()))
            _time.sleep(0.1)                               # 例如下載
            with lock:
                events.append(("end", name, _time.monotonic()))

    threads = [threading.Thread(target=request, args=(n,)) for n in "abc"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    starts = [e for e in events if e[0] == "start"]
    ends = [e for e in events if e[0] == "end"]
    assert [e[0] for e in events] == ["start", "end"] * 3   # 不會同時連
    for prev_end, nxt in zip(ends, starts[1:]):
        assert nxt[2] - prev_end[2] >= 0.045                # 前一次「結束」後才開始算間隔


def test_inner_pool_lines_keep_video_tag():
    import io
    raw = io.StringIO()
    out = vsong.TaggedOutput(raw)
    real = sys.stdout
    sys.stdout = out
    try:
        out.set_tag("[2/5] ")
        with vsong.ThreadPoolExecutor(max_workers=3) as pool:
            for f in [pool.submit(vsong.keep_tag(lambda k: print(f"\r  辨識 {k}/3：歌{k}", flush=True)), k)
                      for k in (1, 2, 3)]:
                f.result()
        out.end()
    finally:
        sys.stdout = real
    lines = raw.getvalue().splitlines()
    assert sorted(lines) == ["[2/5] 辨識 1/3：歌1", "[2/5] 辨識 2/3：歌2", "[2/5] 辨識 3/3：歌3"]


def test_youtube_id_only_youtube():
    assert vsong.youtube_id("https://www.openrec.tv/live/abcdefghijk") is None
    assert vsong.youtube_id("https://example.com/watch?v=aphAfV1Ge8c") is None
    assert vsong.youtube_id("https://www.youtube.com/watch?v=aphAfV1Ge8cXYZ") is None   # ID 只有 11 碼
    assert vsong.youtube_id("https://m.youtube.com/watch?v=aphAfV1Ge8c") == "aphAfV1Ge8c"


def test_reuse_title_finds_merged_song(tmp_path):
    tl = tmp_path / "timeline.json"
    tl.write_text(json.dumps({"songs": [{"start": 2540, "end": 2925, "title": "レオ", "by": "manual"}]}),
                  encoding="utf-8")
    # 之前合併成一首（42:20–48:45），偵測還是切成兩段：兩段都要認得
    assert vsong.reuse_title(tl, 2540, 2781)["title"] == "レオ"
    assert vsong.reuse_title(tl, 2804, 2925)["title"] == "レオ"
    assert vsong.reuse_title(tl, 2900, 3100) is None                      # 大半在外面的不算


def test_merge_same_titles_only_merges_ai_results():
    segs = [(0, 100), (110, 200), (300, 400), (410, 500), (600, 700), (710, 800)]
    names = [{"title": "安可曲", "by": "setlist"}, {"title": "安可曲", "by": "setlist"},   # 歌單寫了兩次：真的唱兩次
             {"title": "Lemon", "artist": "米津玄師", "by": "gemini"},
             {"title": "Lemon", "artist": "Beyoncé", "by": "gemini"},                     # 原唱不同：不同首
             {"title": "レオ", "by": "gemini"}, {"title": "レオ", "by": "gemini"}]
    got, _ = vsong.merge_same_titles(segs, names)
    assert got == [(0, 100), (110, 200), (300, 400), (410, 500), (600, 800)]


@pytest.mark.parametrize("title,artist,results,want", [
    ("Lemon", None, [("Lemonade", "X", 200)], None),                         # 別首歌
    ("春泥棒", "ヨルシカ", [("春", "ヨルシカ", 100), ("春泥棒", "誰か", 300)], 300),  # 同名優先於同原唱
    ("プロポーズ", "なとり", [("プロポーズ (TV size)", "なとり", 90), ("プロポーズ", "なとり", 178)], 178),
    ("アスノヨゾラ哨戒班", None, [("アスノヨゾラ哨戒班 (feat. IA)", "Orangestar", 177)], 177),  # 版本字樣可以
    ("★", None, [("Lemon", "X", 200)], None),                               # 歌名正規化後是空的
])
def test_itunes_matching(monkeypatch, title, artist, results, want):
    payload = {"results": [{"trackName": n, "artistName": a, "trackTimeMillis": t * 1000} for n, a, t in results]}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(vsong.urllib.request, "urlopen", lambda *a, **k: Resp())
    monkeypatch.setattr(vsong, "ITUNES_GAP", 0.0)
    assert vsong._itunes_lookup(title, artist) == want


# ───────────────────────── 講話不要被當成歌、太長的段落切開 ─────────────────────────
def test_loud_talk_is_not_a_song():
    # 0–300 大聲聊天（邊放 BGM），300–540 唱歌；聊天和唱歌一樣大聲
    sc = make_scores([(300, 540, KARAOKE)], n=200)
    sc[:100] = TALK_BGM
    loud = make_loud([(0, 540, -18)], n=200)
    segs = vsong.detect(sc, loud=loud)
    assert len(segs) == 1 and abs(segs[0][0] - 300) <= 6


def test_short_shout_inside_song_keeps_loudness_bonus():
    sc = make_scores([(60, 300, KARAOKE)], n=120)
    t = np.arange(120) * vsong.HOP + vsong.WIN / 2
    sc[(t >= 150) & (t < 159), 2] = 0.6                       # 歌裡喊了一聲（9 秒）
    loud = make_loud([(60, 300, -18)], n=120)
    assert len(vsong.detect(sc, loud=loud)) == 1              # 不會因為一聲喊叫被切斷


def test_split_long_at_quiet_gap():
    loud = make_loud([(0, 900, -18), (448, 454, -30)], n=300)  # 兩首連著唱，中間 6 秒空檔
    got = vsong.split_long([(0, 900)], loud)
    assert len(got) == 2 and 440 <= got[0][1] <= 460
    assert vsong.split_long([(0, 300)], loud) == [(0, 300)]  # 不長的不切
    flat = make_loud([(0, 900, -18)], n=300)
    assert vsong.split_long([(0, 900)], flat) == [(0, 900)]  # 找不到空檔就不切


def test_trim_talk_tail():
    sc = make_scores([(60, 240, SONG)], n=120)                # 240 之後是聊天（CHAT 的 speech 0.8）
    got = vsong.trim_talk_tail([(60, 330)], sc)
    assert 240 <= got[0][1] <= 246
    assert vsong.trim_talk_tail([(60, 200)], sc) == [(60, 200)]


def test_extend_outro_keeps_accompaniment_while_talking():
    # 60–240 唱；240–270 一邊講話一邊還有伴奏（尾奏）；270 之後只剩講話、音量掉下去
    sc = np.zeros((120, 3), dtype=np.float32)
    sc[19:79] = (0.8, 0.6, 0.05)
    sc[79:89] = (0.0, 0.6, 0.7)
    sc[89:] = (0.0, 0.05, 0.8)
    loud = np.full(120, -30.0)
    loud[19:89] = -20.0
    got = vsong.extend_outro([(60, 240)], sc, loud)
    assert 265 <= got[0][1] <= 272
    assert vsong.extend_outro([(60, 240)], sc, loud, fixed={0}) == [(60, 240)]   # 手動歌單寫了結束時間
    # 不超過下一首、最多延伸 OUTRO_MAX 秒
    assert vsong.extend_outro([(60, 240), (255, 300)], sc, loud)[0] == (60, 255)
    sc[79:] = (0.0, 0.6, 0.7)
    loud[:] = -20.0
    assert vsong.extend_outro([(60, 240)], sc, loud)[0][1] <= 240 + vsong.OUTRO_MAX
    # 伴奏小聲很多（BGM）就不延伸
    loud[79:] = -32.0
    assert vsong.extend_outro([(60, 240)], sc, loud) == [(60, 240)]
    assert vsong.extend_outro([(60, 240)], sc, None) == [(60, 240)]


def test_extend_intro_keeps_intro_played_while_talking():
    # 0–60 只有講話；60–90 邊講話邊放前奏；90–240 唱
    sc = np.zeros((90, 3), dtype=np.float32)
    sc[:19] = (0.0, 0.05, 0.8)
    sc[19:29] = (0.0, 0.6, 0.7)
    sc[29:79] = (0.8, 0.6, 0.05)
    loud = np.full(90, -30.0)
    loud[19:79] = -20.0
    got = vsong.extend_intro([(90, 240)], sc, loud)
    assert 58 <= got[0][0] <= 65
    assert vsong.extend_intro([(90, 240)], sc, loud, fixed={0}) == [(90, 240)]           # 歌單的開始時間不動
    assert vsong.extend_intro([(0, 70), (90, 240)], sc, loud)[1][0] == 70                # 不超過上一首


def test_detect_merges_across_interlude_talk_while_music_continues():
    # 60–180 唱、180–198 間奏裡講話（伴奏沒斷、音量沒掉）、198–300 繼續唱 → 一首
    sc = make_scores([(60, 180, SONG), (198, 300, SONG)], n=120)
    sc[59:65] = (0.0, 0.55, 0.6)
    loud = make_loud([(60, 300, -20)], n=120)
    assert len(vsong.detect(sc, max_gap=0, loud=loud)) == 1
    sc[59:65, 1] = 0.05                                                                # 伴奏停了：兩首
    assert len(vsong.detect(sc, max_gap=0, loud=loud)) == 2


@needs_ffmpeg
def test_process_skips_locked_stream(tmp_path, monkeypatch):
    out = tmp_path / "out"
    d = out / "Ch" / "aphAfV1Ge8c"
    write_stream(d, "aphAfV1Ge8c", "20250101", [("A", 0, 30)])
    tl = json.loads((d / "timeline.json").read_text(encoding="utf-8"))
    (d / "timeline.json").write_text(json.dumps({**tl, "locked": True}, ensure_ascii=False), encoding="utf-8")
    before = (d / "timeline.json").read_text(encoding="utf-8")
    monkeypatch.setattr(vsong, "download", lambda *a, **k: pytest.fail("勾了「已手動處理」就不用下載"))
    monkeypatch.setattr(vsong, "analyze", lambda *a, **k: pytest.fail("也不用分析"))
    args = types.SimpleNamespace(out=str(out), cookies_from_browser=None, channel=None, reanalyze=True,
                                 threshold=0.5, min_len=60, max_gap=8, identify=False, merge=False, batch=4)
    vsong.process("https://www.youtube.com/watch?v=aphAfV1Ge8c", args)
    assert (d / "timeline.json").read_text(encoding="utf-8") == before
    assert vsong.locked_dir("https://www.youtube.com/watch?v=2omY8cYhWpU", out) is None


@needs_ffmpeg
def test_setlist_onset_moves_late_start_back_to_intro(tmp_path):
    import wave
    sr = 8000
    t = np.arange(40 * sr) / sr
    rng = np.random.default_rng(0)
    x = 0.2 * rng.standard_normal(len(t))
    talk = (t < 10) & ((t % 0.5) < 0.3)                      # 0–10 秒講話（字和字中間會停）
    x[(t < 10) & ~talk] = 0
    x[(t >= 10) & (t < 10.6)] = 0                            # 伴奏音軌開始前的空隙
    src = tmp_path / "a.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    n = 12
    sc = np.zeros((n, 3), np.float32)
    sc[3:, 1] = 0.6                                          # 10.6 秒之後一直有伴奏
    loud = np.full(n, -20.0)
    # 歌單標在 16 秒（前奏切掉 5 秒多）→ 往前到伴奏開始前一點點
    assert 9.9 <= vsong.setlist_onset(src, sc, loud, 16.0, 36.0, 0.0) <= 10.6
    # 歌單標在伴奏開始那一刻：對的，只對齊到開始前一點點
    assert 9.9 <= vsong.setlist_onset(src, sc, loud, 10.6, 36.0, 0.0) <= 10.6
    # 清唱起頭（10.6～19 秒沒有伴奏）、歌單標在伴奏進來的 19 秒：照歌單，不會亂動
    sc[3:6, 1] = 0.05
    assert vsong.setlist_onset(src, sc, loud, 19.0, 36.0, 0.0) == 19.0
    # 上一首 13 秒才結束：空隙在上一首裡面，不往前
    sc[3:, 1] = 0.6
    assert vsong.setlist_onset(src, sc, loud, 16.0, 36.0, 13.0) == 16.0


@needs_ffmpeg
def test_align_end_to_where_accompaniment_dies_out(tmp_path):
    import wave
    sr = 8000
    t = np.arange(60 * sr) / sr
    x = 0.2 * np.random.default_rng(0).standard_normal(len(t))
    x[(t >= 40.3) & (t < 41)] = 0                           # 40.3 秒伴奏結束（安靜）
    x[(t >= 41) & ((t % 0.5) >= 0.3)] = 0                   # 之後是講話（字和字中間會停）
    src = tmp_path / "a.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    # 粗估 42 秒（晚了）→ 對齊到 40.3 + 餘音
    assert 40.3 <= vsong.align_end(src, 0, 42.0, 100) <= 41.3
    # 粗估 39 秒（早了）→ 往後對齊
    assert 40.3 <= vsong.align_end(src, 0, 39.0, 100) <= 41.3
    # 下一首 39.5 秒就開始：安靜在那之後，不動
    assert vsong.align_end(src, 0, 39.0, 39.5) == 39.0
    # 最多只往前 END_MAX_EARLY 秒
    assert vsong.align_end(src, 0, 45.0, 100) >= 45.0 - vsong.END_MAX_EARLY


@needs_ffmpeg
def test_check_song_flags_uncertain_cut_points(tmp_path, monkeypatch):
    import wave
    sr = 8000
    t = np.arange(120 * sr) / sr
    x = 0.2 * np.random.default_rng(0).standard_normal(len(t))
    x[((t >= 9.5) & (t < 10)) | ((t >= 100) & (t < 101))] = 0     # 10 秒伴奏開始、100 秒伴奏結束
    d = tmp_path / "Ch" / "v1"
    d.mkdir(parents=True)
    with wave.open(str(d / "source.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    sc = np.zeros((40, 3), np.float32)
    sc[:, 1] = 0.6
    assert vsong.check_song(d / "source.wav", sc, 9.5, 100.8) == []                 # 切在安靜的地方：沒問題
    assert vsong.check_song(d / "source.wav", sc, 30.0, 70.0) == ["end", "start"]    # 切在伴奏中間
    sc[20:23, 1] = 0.05                                                               # 60～69 秒沒伴奏
    assert vsong.check_song(d / "source.wav", sc, 9.5, 100.8, orig_len=40) == ["gap", "long"]
    np.save(d / "scores.npy", sc)
    tl = {"source": "source.wav", "songs": [{"start": 9.5, "end": 100.8}, {"start": 30, "end": 70},
                                            {"start": 30, "end": 70, "ok": True}]}
    assert vsong.check_timeline(d, tl) == 2 and tl["songs"][1]["check"] and "check" not in tl["songs"][2]
    (d / "timeline.json").write_text(json.dumps({**tl, "title": "x"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["vsong", "--check", "--out", str(tmp_path)])
    vsong.main()                                                                      # 只檢查、不重新處理
    saved = json.loads((d / "timeline.json").read_text(encoding="utf-8"))
    assert [bool(s.get("check")) for s in saved["songs"]] == [True, True, False]
    (d / "timeline.json").write_text(json.dumps({**saved, "locked": True}), encoding="utf-8")
    vsong.main()                                                                      # 勾了已手動處理的略過
    assert json.loads((d / "timeline.json").read_text(encoding="utf-8"))["songs"] == saved["songs"]


def test_setlist_duplicate_timestamp_parenthetical_and_emoji_codes():
    text = ("5:07 5:07\u3000アウターサイエンス (Outer Science)\n(25:39 夏恋花火)\n1:29:12 （風になる）\n"
            "2:12:29 憂、燦々:virtualhug:\n12:30 3:33 / Junky")
    assert vsong.parse_setlist(text) == [(307, "アウターサイエンス", None), (7949, "憂、燦々", None), (750, "3:33", "Junky")]


def test_apply_setlist_follows_song_across_interlude_and_drops_talk_entries():
    # A：60–150 唱、150–170 間奏時講了幾句、170–300 繼續唱 → 整首都是 A
    # 「お便り」：400–600 一直在講話 → 不是歌
    sc = make_scores([(60, 150, SONG), (150, 170, CHAT), (170, 300, SONG), (700, 900, SONG)], n=400)
    segs = vsong.widen(vsong.detect(sc, max_gap=0), sc)
    entries = [{"t": 60, "title": "A", "artist": None}, {"t": 400, "title": "お便り紹介", "artist": None},
               {"t": 700, "title": "B", "artist": None}]
    # 有原曲長度（240 秒）：往後接到 300 比較接近原曲 → 整首都是 A
    got, names = vsong.apply_setlist(entries, segs, sc, max_gap=0, length_of=lambda t, a: 240 if t == "A" else None)
    assert [n and n["title"] for n in names] == ["A", "B"]
    assert got[0][0] == 60 and got[0][1] >= 298
    # 原曲很短（120 秒）：往後接到 300 超過原曲 60 秒以上，相信偵測結果的結尾
    got, _ = vsong.apply_setlist(entries, segs, sc, max_gap=0, length_of=lambda t, a: 120 if t == "A" else None)
    assert got[0][0] == 60 and got[0][1] < 170
    # 沒有原曲長度：偵測結果常在間奏斷開，相信往後接的結果
    got, _ = vsong.apply_setlist(entries, segs, sc, max_gap=0)
    assert got[0][0] == 60 and got[0][1] >= 298


def test_choose_setlist_ignores_highlight_block_after_setlist():
    segs = [(281, 456), (785, 987), (1277, 1620), (1907, 2187)]
    text = ("4:38 プロポーズ\n13:13 Overdose\n21:22 君の知らない物語\n31:53 lulu.\n\n"
            "5:00 ここ好き\n14:00 かわいい\n22:00 草")              # 後面附的精華時刻：時間倒回去
    got = vsong.choose_setlist([text], segs)
    assert [e[1] for e in got] == ["プロポーズ", "Overdose", "君の知らない物語", "lulu."]


def test_parse_setlist_more_formats():
    assert vsong.parse_setlist("05:07 5:07 アウターサイエンス") == [(307, "アウターサイエンス", None)]
    assert vsong.parse_setlist("恋愛裁判　13:05\n逆光　3:50:10") == [(785, "恋愛裁判", None), (13810, "逆光", None)]
    assert vsong.parse_setlist("3:00 群青:_akubiBii: / YOASOBI") == [(180, "群青", "YOASOBI")]
    assert vsong.parse_setlist("3:00 Re:Re: / ASIAN KUNG-FU GENERATION")[0][1] == "Re:Re:"
    assert vsong.parse_setlist("1:06:59 高評価1500達成\n1:27:55 お手紙") == []
    assert [t for _, t, _ in vsong.parse_setlist("27:41 《4曲目》 ロメオ\n33:44 【5曲目】心という名の不可解")] == \
        ["ロメオ", "心という名の不可解"]


@needs_ffmpeg
def test_manual_setlist_is_trusted_and_survives_reidentify(tmp_path, monkeypatch):
    audio = tmp_path / "歌回.m4a"
    make_audio(audio, 400)
    # 偵測只抓到 60–180；手動歌單寫的是 0:30 和 3:00（跟偵測對不太上，留言來的歌單會被拒絕）
    monkeypatch.setattr(vsong, "analyze", lambda wav, batch: make_scores([(60, 180, SONG), (180, 330, QUIET)], n=132))
    monkeypatch.setattr(vsong, "fetch_setlist", lambda *a, **k: pytest.fail("手動歌單不用讀留言"))
    monkeypatch.setattr(vsong, "identify", lambda src, segs, *a, **k: [{"title": "Gemini 猜的"} for _ in segs])
    args = types.SimpleNamespace(out=str(tmp_path / "out"), cookies_from_browser=None, channel=None, reanalyze=False,
                                 threshold=0.5, min_len=60, max_gap=8, identify=False, merge=False, batch=4)
    vsong.process(str(audio), args)
    tl_p = next((tmp_path / "out").glob("*/*/timeline.json"))
    tl = json.loads(tl_p.read_text(encoding="utf-8"))
    tl["setlist"] = {"texts": ["0:30 手動A / 原唱\n3:00 手動B"], "entries": [], "manual": True}
    tl_p.write_text(json.dumps(tl, ensure_ascii=False), encoding="utf-8")
    args.identify, args.reidentify = True, True
    vsong.process(str(audio), args)
    tl = json.loads(tl_p.read_text(encoding="utf-8"))
    assert [(s["start"], s["title"], s["by"]) for s in tl["songs"]] == [(30, "手動A", "setlist"), (180, "手動B", "setlist")]
    assert tl["setlist"]["manual"] is True and len(tl["setlist"]["entries"]) == 2


# ───────────────────────── Holodex ─────────────────────────
def fake_holodex(monkeypatch, payload, calls=None):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return json.dumps(payload).encode("utf-8")

    def urlopen(req, timeout=20):
        if calls is not None:
            calls.append(req)
        return Resp()

    monkeypatch.setattr(vsong.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(vsong, "HOLODEX_GAP", 0.0)


def test_clean_holodex_name():
    assert vsong.clean_holodex_name("星に願いを / Hoshi Ni Negaiwo") == "星に願いを"
    assert vsong.clean_holodex_name("ヒバナ (Hibana)") == "ヒバナ"
    assert vsong.clean_holodex_name("レクイエム (feat. 星街すいせい) / Requiem (feat. Hoshimachi Suisei)") == "レクイエム"
    assert vsong.clean_holodex_name("LOSER / Loser") == "LOSER"
    assert vsong.clean_holodex_name("サムライハート(Some Like It Hot!!) - New Version -") == "サムライハート"
    assert vsong.clean_holodex_name("Lemon (TV Size)") == "Lemon"
    assert vsong.clean_holodex_name("Re:Re:") == "Re:Re:"


def test_fetch_holodex(monkeypatch):
    calls = []
    fake_holodex(monkeypatch, {"songs": [{"name": "LOSER / Loser", "original_artist": "米津玄師", "start": 300, "end": 540},
                                         {"name": "怪獣の花唄", "original_artist": "Vaundy", "start": 60, "end": 280}],
                               "comments": [{"message": "4:38 プロポーズ"}]}, calls)
    got = vsong.fetch_holodex("aphAfV1Ge8c", "KEY")
    assert got["songs"] == [{"t": 60.0, "title": "怪獣の花唄", "artist": "Vaundy", "end": 280.0},
                            {"t": 300.0, "title": "LOSER", "artist": "米津玄師", "end": 540.0}]
    assert got["comments"] == ["4:38 プロポーズ"]
    req = calls[0]
    assert req.get_header("X-apikey") == "KEY" and "vsong" in req.get_header("User-agent")
    assert "aphAfV1Ge8c" in req.full_url and "include=songs" in req.full_url and "c=1" in req.full_url


@needs_ffmpeg
def test_holodex_songs_are_used_and_checked_once(tmp_path, monkeypatch):
    pages = {"https://www.youtube.com/watch?v=aphAfV1Ge8c":
             {"id": "aphAfV1Ge8c", "title": "歌枠", "channel": "Aちゃん", "channel_id": "UCa", "upload_date": "20261020"}}
    fake_ytdlp(monkeypatch, pages,
               on_download=lambda opts, info: make_audio(opts["outtmpl"].replace("%(ext)s", "m4a"), 600))
    monkeypatch.setattr(vsong, "analyze", lambda wav, batch: make_scores([(60, 180, SONG), (300, 420, SONG)], n=199))
    monkeypatch.setenv("HOLODEX_API_KEY", "KEY")
    monkeypatch.setattr(vsong, "fetch_setlist", lambda *a, **k: pytest.fail("Holodex 有歌單就不用讀 YouTube 留言"))
    monkeypatch.setattr(vsong, "identify", lambda *a, **k: pytest.fail("不用 Gemini"))
    calls = []
    fake_holodex(monkeypatch, {"songs": [{"name": "歌A", "original_artist": "原唱", "start": 62, "end": 175},
                                         {"name": "歌B / Uta B", "original_artist": None, "start": 300, "end": 410}],
                               "comments": []}, calls)
    args = types.SimpleNamespace(out=str(tmp_path / "out"), cookies_from_browser=None, channel=None, reanalyze=False,
                                 threshold=0.5, min_len=60, max_gap=8, identify=True, merge=False, batch=4)
    vsong.process("https://www.youtube.com/watch?v=aphAfV1Ge8c", args)
    tl_p = next((tmp_path / "out").glob("*/aphAfV1Ge8c/timeline.json"))
    tl = json.loads(tl_p.read_text(encoding="utf-8"))
    # 照 Holodex 的開始時間；結尾從 Holodex 的結束時間再延到伴奏結束（歌其實唱到 180）
    assert [(s["start"], s["title"], s["by"]) for s in tl["songs"]] == [(62, "歌A", "setlist"), (300, "歌B", "setlist")]
    assert [175 <= tl["songs"][0]["end"] <= 182, 410 <= tl["songs"][1]["end"] <= 422] == [True, True]
    assert tl["setlist"]["source"] == "holodex" and tl["holodex_checked"] is True
    vsong.process("https://www.youtube.com/watch?v=aphAfV1Ge8c", args)   # 重跑：查過了，不再查
    assert len(calls) == 1
    tl = json.loads(tl_p.read_text(encoding="utf-8"))
    assert tl["setlist"]["source"] == "holodex" and [s["title"] for s in tl["songs"]] == ["歌A", "歌B"]


@needs_ffmpeg
def test_holodex_comments_fill_missing_setlist(tmp_path, monkeypatch):
    pages = {"https://www.youtube.com/watch?v=aphAfV1Ge8c":
             {"id": "aphAfV1Ge8c", "title": "歌枠", "channel": "Aちゃん", "channel_id": "UCa", "upload_date": "20261020"}}
    fake_ytdlp(monkeypatch, pages,
               on_download=lambda opts, info: make_audio(opts["outtmpl"].replace("%(ext)s", "m4a"), 600))
    monkeypatch.setattr(vsong, "analyze", lambda wav, batch: make_scores([(60, 180, SONG), (300, 420, SONG)], n=199))
    monkeypatch.setenv("HOLODEX_API_KEY", "KEY")
    monkeypatch.setattr(vsong, "fetch_setlist", lambda *a, **k: pytest.fail("Holodex 留言夠了就不用讀 YouTube"))
    fake_holodex(monkeypatch, {"songs": [], "comments": [{"message": "かわいい"},
                                                         {"message": "1:00 留言歌A / 原唱\n5:00 留言歌B"}]})
    args = types.SimpleNamespace(out=str(tmp_path / "out"), cookies_from_browser=None, channel=None, reanalyze=False,
                                 threshold=0.5, min_len=60, max_gap=8, identify=True, merge=False, batch=4)
    vsong.process("https://www.youtube.com/watch?v=aphAfV1Ge8c", args)
    tl = json.loads(next((tmp_path / "out").glob("*/aphAfV1Ge8c/timeline.json")).read_text(encoding="utf-8"))
    assert [s["title"] for s in tl["songs"]] == ["留言歌A", "留言歌B"]
    assert tl["setlist"]["source"] == "comments" and tl["setlist"]["texts"] == ["1:00 留言歌A / 原唱\n5:00 留言歌B"]


# ───────────────────────── 時間戳錯誤的歌單 ─────────────────────────
def test_setlist_plausible_rejects_wrong_timestamps():
    segs = [(281, 456), (785, 987), (1277, 1620), (1907, 2187), (2500, 2760)]
    right = [(278, "A", None), (793, "B", None), (1281, "C", None), (1911, "D", None)]
    assert vsong.setlist_plausible(right, segs)
    # 別場的歌單：時間大多落在歌的中間
    wrong = [(400, "X", None), (1000, "Y", None), (1500, "Z", None), (2100, "W", None)]
    assert not vsong.setlist_plausible(wrong, segs)
    # 整份偏移了 3 分鐘（例如從剪輯版抄來的）
    assert not vsong.setlist_plausible([(t + 180, ti, ar) for t, ti, ar in right], segs)
    assert not vsong.setlist_plausible(right[:1], segs)                  # 只有一首：不夠判斷


def test_setlist_plausible_back_to_back_songs_use_quiet_gaps():
    # 偵測把三首連著唱的歌黏成一段（0–900），但歌和歌之間有 6 秒小聲的空檔
    loud = make_loud([(0, 900, -18), (298, 304, -30), (598, 604, -30)], n=300)
    segs = [(0, 900)]
    entries = [(2, "A", None), (302, "B", None), (601, "C", None)]
    assert vsong.setlist_plausible(entries, segs, loud)
    assert not vsong.setlist_plausible([(150, "A", None), (450, "B", None), (750, "C", None)], segs, loud)


def test_choose_setlist_rejects_unrelated_single_comment():
    segs = [(281, 456), (785, 987), (1277, 1620), (1907, 2187)]
    unrelated = "5:30 プロポーズ\n16:40 Overdose\n26:00 君の知らない物語\n35:00 lulu."   # 別場的時間
    assert vsong.choose_setlist([unrelated], segs) == []
    real = "4:38 プロポーズ\n13:13 Overdose\n21:22 君の知らない物語\n31:53 lulu."
    assert len(vsong.choose_setlist([unrelated, real], segs)) == 4


def test_holodex_tags_with_wrong_times_are_ignored(tmp_path, monkeypatch):
    pages = {"https://www.youtube.com/watch?v=aphAfV1Ge8c":
             {"id": "aphAfV1Ge8c", "title": "歌枠", "channel": "Aちゃん", "channel_id": "UCa", "upload_date": "20261020"}}
    if shutil.which("ffmpeg") is None:
        pytest.skip("需要 ffmpeg")
    fake_ytdlp(monkeypatch, pages,
               on_download=lambda opts, info: make_audio(opts["outtmpl"].replace("%(ext)s", "m4a"), 600))
    monkeypatch.setattr(vsong, "analyze", lambda wav, batch: make_scores([(60, 180, SONG), (300, 420, SONG)], n=199))
    monkeypatch.setenv("HOLODEX_API_KEY", "KEY")
    monkeypatch.setattr(vsong, "fetch_setlist", lambda *a, **k: [])
    monkeypatch.setattr(vsong, "identify", lambda src, segs, *a, **k: [None for _ in segs])
    fake_holodex(monkeypatch, {"songs": [{"name": "別場A", "start": 120, "end": 200},      # 時間都在歌的中間
                                         {"name": "別場B", "start": 360, "end": 500}], "comments": []})
    args = types.SimpleNamespace(out=str(tmp_path / "out"), cookies_from_browser=None, channel=None, reanalyze=False,
                                 threshold=0.5, min_len=60, max_gap=8, identify=True, merge=False, batch=4)
    vsong.process("https://www.youtube.com/watch?v=aphAfV1Ge8c", args)
    tl = json.loads(next((tmp_path / "out").glob("*/aphAfV1Ge8c/timeline.json")).read_text(encoding="utf-8"))
    assert "setlist" not in tl and all(s["title"] is None for s in tl["songs"])

