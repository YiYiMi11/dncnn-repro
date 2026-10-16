"""下载复现 DnCNN 所需的数据。

1. 训练集：BSDS500 的 train(200) + val(200) = 400 张原图
   （即文献/TNRD 所说的 BSD400，构成上等价，且与测试集零重叠）
2. 测试集：DnCNN 作者官方仓库中的 Set12（12 张彩色）与 BSD68（68 张彩色）

用法:
    python download_data.py
"""

import argparse
import base64
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# BIDS/BSDS500 是 Berkeley BSDS500（2013 版）的官方镜像
BSDS_REPO = "https://api.github.com/repos/BIDS/BSDS500/contents/BSDS500/data/images"
# DnCNN 作者 cszn 官方测试集
DNCNN_REPO = "https://api.github.com/repos/cszn/DnCNN/contents/testsets"

UA = {"User-Agent": "Mozilla/5.0 (dncnn-repro)"}


def _ssl_ctx():
    """macOS 上 python.org 官方解释器默认找不到 CA 根证书，能用 certifi 就用它。"""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CTX = _ssl_ctx()





TIMEOUT = int(os.environ.get("DNCNN_DL_TIMEOUT", 120))


def _curl(url, timeout=None):
    """调系统 curl（macOS 自带，证书链正常）。加硬超时，避免网络异常时线程永久阻塞。"""
    import subprocess
    timeout = timeout or TIMEOUT
    p = subprocess.run(
        ["curl", "-fsSL", "--max-time", str(timeout), "--retry", "1",
         "--retry-delay", "1", "--connect-timeout", "20", url],
        capture_output=True, timeout=timeout + 30,
    )
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(
            f"curl exit {p.returncode}: {p.stderr.decode(errors='ignore')[:160]}")
    return p.stdout


def _headers():
    h = dict(UA)
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


# GitHub 加速代理：raw.githubusercontent.com 解析不了时的备用通道。
# 依次尝试，取第一个能用的（可用 DNCNN_PROXY 指定；设空串可禁用）。
PROXY_CANDIDATES = [
    os.environ.get("DNCNN_PROXY", "https://ghproxy.net/"),
    "https://gh-proxy.com/",
    "https://ghfast.top/",
    "https://mirror.ghproxy.com/",
    "https://gh.llkk.cc/",
]
_PROXY_OK = []


def _try_proxies(url, timeout=None):
    """依次尝试各 GitHub 代理，返回图片字节。

    上次成功过的代理排在最前面省时间，但**失败后要把它从优先列表里剔除**——
    否则一旦被记为"好"，之后它挂掉时会被永远跳过，整批下载就静默失败了。
    """
    if "raw.githubusercontent.com" not in url:
        return None
    for p in list(_PROXY_OK):
        try:
            data = _curl(p.rstrip("/") + "/" + url, timeout or 60)
            if _looks_like_image(data):
                return data
            _PROXY_OK.remove(p)
        except Exception:  # noqa: BLE001
            _PROXY_OK.remove(p)
    for p in PROXY_CANDIDATES:
        if not p or p in _PROXY_OK:
            continue
        try:
            data = _curl(p.rstrip("/") + "/" + url, timeout or 60)
            if _looks_like_image(data):
                _PROXY_OK.append(p)
                print(f"  [proxy] 加速通道可用: {p}")
                return data
        except Exception:  # noqa: BLE001
            continue
    return None


def _urllib(url, timeout=None):
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout or TIMEOUT, context=SSL_CTX) as r:
        return r.read()


def _gh_api(url, timeout=None):
    """用 GitHub contents API 取文件内容（base64）。

    有些网络环境解析不了 raw.githubusercontent.com，但 api.github.com 是通的。
    这个接口能直接返回文件内容（约 1MB 以内，图片足够），因此作为兜底通道。
    """
    t = timeout or TIMEOUT
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=t, context=SSL_CTX) as r:
        d = json.load(r)
    if "content" not in d:
        raise RuntimeError(f"contents API 无内容: {str(d)[:120]}")
    return base64.b64decode(d["content"])


def _rate_limit_error(e):
    return "403" in str(e) and "rate limit" in str(e).lower()


def _wait_for_rate_limit():
    """读到配额重置时间就等着，不够就提示用户去配 token。"""
    try:
        req = urllib.request.Request("https://api.github.com/rate_limit", headers=_headers())
        with urllib.request.urlopen(req, timeout=20, context=SSL_CTX) as r:
            d = json.load(r)["resources"]["core"]
    except Exception:  # noqa: BLE001
        return False
    rem, reset = d.get("remaining", 0), d.get("reset", 0)
    if rem > 0:
        return True
    wait = max(0, reset - int(time.time())) + 5
    if wait > 900:
        print(f"  [!!] GitHub API 配额已用完（约 {wait // 60} 分钟后重置）。")
        print("       匿名配额只有 60 次/小时，下载 480 个文件肯定不够用。")
        print("       请配一个 token（免费）：")
        print("         1) 浏览器登录 https://github.com/settings/tokens")
        print("         2) 生成一个 classic token，只需勾选 public_repo")
        print("         3) export GITHUB_TOKEN=ghp_xxx   （写进 ~/.zshrc 可持久化）")
        print(f"       然后重跑本脚本即可，已下好的文件不会重复下载。")
        return False
    print(f"  [info] GitHub 配额用完，等待 {wait} 秒后自动继续 ...")
    while time.time() < reset + 5:
        time.sleep(15)
    return True


def _is_gh_api(url):
    return url.startswith("https://api.github.com/repos/")


def _get_json(url):
    """取一个 JSON（目录列表 / 速率信息等）。

    注意这里**不能**走 _gh_api：contents 接口对目录返回的是 JSON 数组，
    对单文件才返回带 base64 content 的对象。这里要的就是 JSON 本身。
    """
    last = None
    for attempt in range(1, 4):
        for fn in (_curl, _urllib):
            try:
                raw = fn(url, 60)
                if raw:
                    return json.loads(raw)
            except Exception as e:  # noqa: BLE001
                last = e
                if _rate_limit_error(e):
                    if _wait_for_rate_limit():
                        continue
                    raise
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"取 JSON 失败 {url}: {type(last).__name__}: {str(last)[:120]}")


IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"\x00\x00\x01\x00", b"BM")


def _looks_like_image(data):
    """确认拿到的是图片字节而不是 JSON/HTML 错误页。"""
    return bool(data) and any(data.startswith(m) for m in IMAGE_MAGIC)


def _fetch(raw_url, api_url=None, timeout=None, retries=3, wait_on_limit=False):
    """取一个文件的内容。

    优先用 raw 地址（可走加速代理，快且不吃 API 配额），
    实在不行再退回 contents 接口（base64，需要 GitHub 配额）。

    注意 contents 接口返回的是 **base64 编码的 JSON 信封**而不是文件本身，
    这种情况必须用 _gh_api 解析，否则会把那段 JSON 文本当成图片存进磁盘。
    """
    last = None
    if raw_url and "raw.githubusercontent.com" in raw_url:
        for attempt in range(1, retries + 1):
            for fn in (lambda u, t: _try_proxies(u, t), _curl, _urllib):
                try:
                    data = fn(raw_url, timeout)
                    if _looks_like_image(data):
                        return data
                    last = RuntimeError(
                        f"不是图片内容(前 32 字节 {bytes(data[:32])!r})")
                except Exception as e:  # noqa: BLE001
                    last = e
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))

    if api_url:
        for attempt in range(1, retries + 1):
            try:
                data = _gh_api(api_url, timeout)
                if _looks_like_image(data):
                    return data
                last = RuntimeError("contents API 返回的不是图片")
            except Exception as e:  # noqa: BLE001
                last = e
                if _rate_limit_error(e):
                    if wait_on_limit and _wait_for_rate_limit():
                        continue
                    raise
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(f"下载失败 {raw_url or api_url}: {type(last).__name__}: "
                       f"{str(last)[:120]}")


def _download(raw_url, api_url, dest):
    """下载一个文件到 dest（先写 .part 再原子替换，中断不会留下半个文件）。"""
    data = _fetch(raw_url, api_url)
    tmp = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return len(data)


_RAW_OK = None


def raw_reachable():
    """探测 raw.githubusercontent.com 是否可用（缓存结果）。

    某些校园网/公司网解析不了 raw 域名，此时改走 GitHub 加速代理。
    """
    global _RAW_OK
    if _RAW_OK is None:
        try:
            import subprocess
            p = subprocess.run(
                ["curl", "-fsI", "--max-time", "8", "--connect-timeout", "5",
                 "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore"],
                capture_output=True, timeout=15)
            _RAW_OK = (p.returncode == 0)
        except Exception:  # noqa: BLE001
            _RAW_OK = False
        if not _RAW_OK:
            print("  [info] raw.githubusercontent.com 直连不可用，"
                  "改用 GitHub 加速通道")
    return _RAW_OK


def fetch_ls(url, exts=(".jpg", ".png", ".jpeg")):
    """递归列出 GitHub API 目录里所有图片 -> [(name, raw_url, api_url)]

    两个 URL 都留着：raw 地址快（可走代理），api 地址是兜底（但吃速率配额）。
    """
    out = []
    for item in _get_json(url):
        if item["type"] == "dir":
            out += fetch_ls(item["url"])
        elif item["name"].lower().endswith(exts):
            out.append((item["name"], item["download_url"], item["url"]))
    return sorted(out)


# --------------------------------------------------------------------------- #
#  目录列举的离线兜底
# --------------------------------------------------------------------------- #
#  列目录必须调 GitHub API，匿名配额只有 60 次/小时，很容易被打满。
# 但 DnCNN 官方测试集的文件名是确定的，可以直接拼出 raw 地址，
# 由代理通道下载，完全不碰 API —— 配额耗尽时这是唯一能走的路。
KNOWN_TESTSETS = {
    "BSD68": ("cszn/DnCNN", "testsets/BSD68",
              [f"test{i:03d}.png" for i in range(1, 69)]),
    "Set12": ("cszn/DnCNN", "testsets/Set12",
              [f"{i:02d}.png" for i in range(1, 13)]),
}


def fetch_ls_known(name):
    """不用 API，直接按已知文件名拼 URL -> [(name, raw_url, None)]"""
    if name not in KNOWN_TESTSETS:
        return []
    repo, path, files = KNOWN_TESTSETS[name]
    return [(f, f"https://raw.githubusercontent.com/{repo}/master/{path}/{f}", None)
            for f in files]


def download_repo(url, dest_dir, exts=(".jpg", ".png", ".jpeg"), workers=4, rounds=5,
                  known=None):
    """并发下载一个目录下的所有图片；网络不稳时多轮补漏（已下载的自动跳过，可反复重跑）。

    known 非空时优先用它列文件（不调 API）；列不出来再退回 API 列举。
    """
    items = fetch_ls_known(known) if known else []
    if items:
        print(f"  [offline] 用内置文件名清单列举（不调 GitHub API），共 {len(items)} 个")
    else:
        try:
            items = fetch_ls(url, exts)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] 列目录失败({str(e)[:90]})")
            return 0
    if not items:
        print(f"  !! {url} 没有列出任何图片，检查网络或仓库结构")
        return 0
    os.makedirs(dest_dir, exist_ok=True)
    total = len(items)

    def have(name):
        p = os.path.join(dest_dir, name)
        return os.path.exists(p) and os.path.getsize(p) > 0

    for rnd in range(1, rounds + 1):
        todo = [(n, u, a) for n, u, a in items if not have(n)]
        if not todo:
            break
        print(f"  第 {rnd}/{rounds} 轮：待下载 {len(todo)}/{total} 张，并发 {workers} ...")
        done = [0]
        lock = threading.Lock()

        def one(job):
            name, dl, api_url = job
            try:
                n = _download(dl, api_url, os.path.join(dest_dir, name))
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] {name}: {str(e)[:110]}")
                return False
            with lock:
                done[0] += 1
                if done[0] % 50 == 0:
                    print(f"    ... {done[0]}/{len(todo)}")
            return bool(n)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, todo))

    ok = sum(1 for it in items if have(it[0]))
    print(f"  -> {dest_dir}: {ok}/{total} 张")
    if ok < total:
        print(f"  !! 还有 {total - ok} 张没下来。多跑几次本脚本即可断点续传；"
              f"网络慢可设 DNCNN_DL_WORKERS=2")
    return ok


def extract_tarball(tgz, root, wanted, exts=(".jpg", ".png", ".jpeg")):
    """从 repo tarball 里挑出需要的文件解到 root/<wanted 下相对路径>。

    wanted 是 tar 内部的路径前缀，例如 "BSDS500/data/images/train"。
    """
    import tarfile
    n = 0
    with tarfile.open(tgz) as tf:
        for m in tf:
            if not m.isfile():
                continue
            if wanted not in m.name or not m.name.lower().endswith(exts):
                continue
            rel = m.name.split(wanted, 1)[1].lstrip("/")      # 相对 wanted 的路径
            dst = os.path.join(root, rel)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                n += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                continue
            with open(dst, "wb") as f:
                f.write(src.read())
            n += 1
    return n


def download_tarball(repo, ref="master", dest_tgz=None, timeout=600):
    """整个仓库打成一个 tarball 下载 —— 一次请求搞定几百个文件，
    既不受 GitHub API 速率限制，也不受 raw.githubusercontent.com 域名限制。"""
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{ref}"
    import subprocess
    import tempfile
    dest_tgz = dest_tgz or os.path.join(
        tempfile.gettempdir(), f"{repo.replace('/', '_')}_{ref}.tgz")
    p = subprocess.run(
        ["curl", "-fsSL", "--max-time", str(timeout), "--connect-timeout", "20",
         "--retry", "3", "-o", dest_tgz, url],
        capture_output=True, timeout=timeout + 60)
    if p.returncode != 0 or not os.path.exists(dest_tgz):
        raise RuntimeError(f"tarball 下载失败 exit={p.returncode}: "
                           f"{p.stderr.decode(errors='ignore')[:160]}")
    print(f"  tarball {os.path.getsize(dest_tgz)/1e6:.1f} MB -> {dest_tgz}")
    return dest_tgz


def verify_dir(path):
    """删掉下载中断留下的损坏文件，否则下次会被当成「已下好」跳过。"""
    from PIL import Image
    bad = []
    for f in list_images(path):
        p = os.path.join(path, f)
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:  # noqa: BLE001
            bad.append(p)
    for p in bad:
        try:
            os.remove(p)
        except OSError:
            pass
    if bad:
        print(f"  [清理] {path} 删掉 {len(bad)} 个损坏文件，将重新下载")
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="数据根目录")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("DNCNN_DL_WORKERS", 4)),
                    help="并发下载线程数；代理不稳时调小到 2（默认 4 即可)")
    ap.add_argument("--tarball", action="store_true",
                    help="用整仓 tarball 下载（推荐：1 次请求，不受 API 限流影响）")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    need = {
        "train": os.path.join(root, "BSDS500/train"),
        "val": os.path.join(root, "BSDS500/val"),
        "Set12": os.path.join(root, "testsets/Set12"),
        "BSD68": os.path.join(root, "testsets/BSD68"),
    }
    want = {"train": 200, "val": 200, "Set12": 12, "BSD68": 68}

    for d in need.values():        # 先清掉损坏文件，避免被当成已下好
        verify_dir(d)

    if args.tarball:
        if all(len(list_images(need[k])) >= want[k] for k in need):
            print("数据已齐，跳过下载。")
        else:
            print("[A] BSDS500 整仓 tarball（一次请求，绕开 API 限流） ...")
            try:
                tgz = download_tarball("BIDS/BSDS500", timeout=1800)
                for split in ("train", "val"):
                    n = extract_tarball(tgz, need[split],
                                        f"BSDS500/data/images/{split}")
                    print(f"  -> BSDS500/{split}: {n} 张")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] tarball 失败({str(e)[:90]})，回退到逐文件下载")

            print("[B] DnCNN 测试集 tarball ...")
            try:
                tgz = download_tarball("cszn/DnCNN", timeout=300)
                for name in ("Set12", "BSD68"):
                    n = extract_tarball(tgz, need[name], f"testsets/{name}")
                    if n == 0:   # 老版本仓库里测试集在 TrainingCodes 下面
                        n = extract_tarball(
                            tgz, need[name],
                            "TrainingCodes/DnCNN_TrainingCodes_DagNN_v1.1/testsets/" + name)
                    print(f"  -> testsets/{name}: {n} 张")
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] tarball 失败({str(e)[:90]})，回退到逐文件下载")

    # 还缺什么就逐文件补（断点续传，已下好的跳过）
    jobs = [
        ("[1] BSDS500 split=train (200 张) ...", f"{BSDS_REPO}/train",
         need["train"], None),
        ("[2] BSDS500 split=val (200 张，与 train 合并即 BSD400) ...",
         f"{BSDS_REPO}/val", need["val"], None),
        ("[3] 官方测试集 Set12 (12 张) ...", f"{DNCNN_REPO}/Set12",
         need["Set12"], "Set12"),
        ("[4] 官方测试集 BSD68 (68 张) ...", f"{DNCNN_REPO}/BSD68",
         need["BSD68"], "BSD68"),
    ]
    todo = [(m, api, dst, kn) for m, api, dst, kn in jobs
            if len(list_images(dst)) < want[os.path.basename(dst)]]
    if todo:
        print("\n以下目录还没齐，逐文件补（已下好的会跳过）：")
        for m, api, dst, kn in todo:
            print(m)
            try:
                download_repo(api, dst, workers=args.workers, known=kn)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                print(f"  [warn] {dst}: {msg[:120]}")
                if "rate limit" in msg.lower():
                    print("  (撞到 GitHub 限流，剩下的等配额恢复或配 token 后再跑)")
    else:
        print("\n所有数据已齐。")

    c = {k: len(list_images(v)) for k, v in need.items()}
    print("\n" + "   ".join(f"{k}: {c[k]}/{want[k]}" for k in
                             ("train", "val", "Set12", "BSD68")))
    if c["Set12"] == 12 and c["BSD68"] == 68:
        print("测试集齐了。下一步： python -m src.train --color --sigma 25"
              f"   (训练图 {c['train'] + c['val']}/400)")
    else:
        print("测试集还没下全。网络好转后重跑本脚本即可续传，"
              "或按提示配 GITHUB_TOKEN / 改用 --tarball。")


def list_images(root):
    if not os.path.isdir(root):
        return []
    return [f for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]


if __name__ == "__main__":
    main()