#!/usr/bin/env python3
"""Check every file in the model catalogs against Hugging Face: the commit exists, the file is in it, and its size
and sha256 match. Needs network, no downloads of model files (small files that Hugging Face does not hash, such as
vocab.txt and config.json, are downloaded and hashed here). Run: python3 tools/verify-catalog.py"""
import hashlib, json, pathlib, re, sys, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = sorted((ROOT / "crates" / "models" / "src").glob("*catalog*.rs"))
FILE = re.compile(r"ModelFile\s*\{(.*?)\n\s*\}", re.S)
URL_RE = re.compile(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)")
trees = {}


def get(url):
    return urllib.request.urlopen(url, timeout=60)


def tree(repo, rev):
    if (repo, rev) not in trees:
        files, url = {}, f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1"
        while url:
            r = get(url)
            for f in json.load(r):
                if f["type"] == "file":
                    files[f["path"]] = (f["size"], (f.get("lfs") or {}).get("oid"))
            url = next((p.split(";")[0].strip(" <>") for p in (r.headers.get("Link") or "").split(",") if 'rel="next"' in p), None)
        trees[(repo, rev)] = files
    return trees[(repo, rev)]


def fields(block):
    name = re.search(r'name:\s*"([^"]+)"', block).group(1)
    url = "".join(re.findall(r'"([^"]*)"', re.search(r"url:\s*(.*?),\s*\n\s*size:", block, re.S).group(1)))
    size = int(re.search(r"size:\s*([\d_]+)", block).group(1).replace("_", ""))
    sha = re.search(r'sha256:\s*"([0-9a-f]+)"', block).group(1)
    return name, url, size, sha


bad = checked = 0
for src in SOURCES:
    for block in FILE.findall(src.read_text()):
        if "sha256: \"" not in block:
            continue  # the struct definition, not a file
        name, url, size, sha = fields(block)
        checked += 1
        m = URL_RE.match(url)
        if not m or not url.endswith(name) and not url.endswith("/" + name):
            print(f"BAD URL   {src.name} {name}: {url}"); bad += 1; continue
        repo, rev, path = m.groups()
        remote = tree(repo, rev).get(path)
        if remote is None:
            print(f"MISSING   {repo}@{rev[:8]} {path}"); bad += 1; continue
        rsize, oid = remote
        if oid is None:  # not stored with LFS: Hugging Face has no sha256 for it, so hash the file itself
            if rsize > 5_000_000:
                print(f"NO HASH   {repo} {path} is {rsize} bytes and has no LFS sha256"); bad += 1; continue
            oid = hashlib.sha256(get(url).read()).hexdigest()
        if rsize != size or oid != sha:
            print(f"MISMATCH  {src.name} {name}: catalog {size} {sha[:12]} / remote {rsize} {oid[:12]}"); bad += 1
print(f"{checked} files checked, {bad} problems")
sys.exit(1 if bad else 0)
