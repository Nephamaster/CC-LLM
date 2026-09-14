"""Collect licensed repository cards, documentation, code, and allowlisted web pages."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from scripts.data_factory.io_utils import iter_jsonl
from scripts.data_factory.text import VERIFIED_LICENSES, canonical_license, normalize_text


DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}
CODE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".go", ".h", ".hpp", ".java", ".js", ".jsx", ".php",
    ".py", ".rb", ".rs", ".scala", ".sh", ".sql", ".swift", ".ts", ".tsx",
}
STRUCTURED_EXTENSIONS = {".json", ".toml", ".xml", ".yaml", ".yml"}
EXCLUDED_PARTS = {".git", "build", "dist", "generated", "node_modules", "third_party", "vendor"}


class ArchiveTooLargeError(ValueError):
    """Raised when a repository archive exceeds the configured limit."""


class MainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipped = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "footer", "noscript"}:
            self.skipped += 1
        elif not self.skipped and tag in {"p", "div", "section", "article", "li", "br", "pre", "code", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "noscript"}:
            self.skipped = max(0, self.skipped - 1)
        elif not self.skipped and tag in {"p", "div", "section", "article", "li", "pre", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skipped:
            self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts)
        lines = [" ".join(line.split()) for line in value.splitlines()]
        return normalize_text("\n".join(line for line in lines if line))


class HttpClient:
    def __init__(self, token: str | None, user_agent: str, delay: float) -> None:
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.delay = delay

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        response = self.session.get(url, timeout=60, **kwargs)
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", "60")))
            response = self.session.get(url, timeout=60, **kwargs)
        response.raise_for_status()
        if self.delay:
            time.sleep(self.delay)
        return response


def _license(item: dict[str, Any], detected: Any = None) -> str:
    declared = canonical_license(item.get("license", detected))
    observed = canonical_license(detected) if detected else declared
    if declared not in VERIFIED_LICENSES or observed not in VERIFIED_LICENSES:
        raise ValueError(f"unverified license: declared={declared!r}, detected={observed!r}")
    if detected and declared != observed:
        raise ValueError(f"license mismatch: declared={declared!r}, detected={observed!r}")
    return declared


def _defaults(item: dict[str, Any], provider: str) -> tuple[str, str]:
    mode = str(item.get("mode", "mixed"))
    if mode == "mixed":
        return "mixed_zh_en", {"github": "github", "huggingface": "huggingface", "web": "web_allowlist"}[provider]
    if mode in {"code", "structured", "math", "hanzi", "chat"}:
        return "supplemental", mode
    raise ValueError(f"unsupported collection mode: {mode}")


def _include_path(path: str, mode: str) -> bool:
    pure = PurePosixPath(path)
    if any(part.lower() in EXCLUDED_PARTS for part in pure.parts) or pure.name.lower().endswith(".min.js"):
        return False
    suffix = pure.suffix.lower()
    if mode == "mixed":
        return suffix in DOC_EXTENSIONS and (pure.name.lower().startswith("readme") or "docs" in {p.lower() for p in pure.parts})
    if mode == "code":
        return suffix in CODE_EXTENSIONS
    if mode == "structured":
        return suffix in STRUCTURED_EXTENSIONS | DOC_EXTENSIONS
    return suffix in DOC_EXTENSIONS


def _chunks(text: str, max_chars: int) -> Iterator[str]:
    text = normalize_text(text)
    lines = text.splitlines()
    chunk: list[str] = []
    size = 0
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if chunk and size + len(line) + 1 > max_chars and not in_fence:
            yield "\n".join(chunk).strip()
            chunk = []
            size = 0
        chunk.append(line)
        size += len(line) + 1
    if chunk:
        yield "\n".join(chunk).strip()


def _doc_id(provider: str, owner: str, revision: str, path: str, chunk_index: int) -> str:
    payload = f"{owner}:{revision}:{path}:{chunk_index}".encode("utf-8")
    return f"{provider}-{hashlib.sha256(payload).hexdigest()[:24]}"


def collect_github(
    item: dict[str, Any],
    client: HttpClient,
    max_file_bytes: int,
    max_archive_bytes: int,
) -> Iterator[dict[str, Any]]:
    repo = str(item["repo"])
    api = f"https://api.github.com/repos/{repo}"
    repository = client.get(api).json()
    detected = client.get(f"{api}/license").json().get("license", {}).get("spdx_id")
    license_name = _license(item, detected)
    revision_name = str(item.get("revision") or repository["default_branch"])
    revision = str(client.get(f"{api}/commits/{revision_name}").json()["sha"])
    mode = str(item.get("mode", "mixed"))
    category, quota_group = _defaults(item, "github")

    archive_limit = int(item.get("max_archive_bytes", max_archive_bytes))
    response = client.get(f"{api}/tarball/{revision}")
    if len(response.content) > archive_limit:
        raise ArchiveTooLargeError(f"GitHub archive exceeds {archive_limit} bytes: {repo}")

    selected = 0
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) < 2:
                continue
            path = PurePosixPath(*parts[1:]).as_posix()
            if not _include_path(path, mode) or member.size > max_file_bytes:
                continue
            if selected >= int(item.get("max_files", 10_000)):
                break
            selected += 1
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            try:
                text = extracted.read().decode("utf-8")
            except UnicodeDecodeError:
                continue
            url = f"https://github.com/{repo}/blob/{revision}/{path}"
            chunks = [normalize_text(text)] if mode == "structured" else _chunks(
                text, int(item.get("max_chars", 16_000))
            )
            for chunk_index, chunk in enumerate(chunks):
                yield {
                    "text": chunk,
                    "source": "github",
                    "repo": repo,
                    "doc_id": _doc_id("github", repo, revision, path, chunk_index),
                    "license": license_name,
                    "url": url,
                    "path": path,
                    "revision": revision,
                    "category": category,
                    "quota_group": quota_group,
                }

def collect_huggingface(item: dict[str, Any], max_file_bytes: int) -> Iterator[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for Hugging Face collection") from error

    repo = str(item["repo"])
    repo_type = str(item.get("repo_type", "model"))
    api = HfApi(token=os.getenv("HF_TOKEN"))
    info = api.repo_info(repo, repo_type=repo_type, revision=item.get("revision"))
    card_data = getattr(info, "card_data", None)
    detected = card_data.get("license") if isinstance(card_data, dict) else getattr(card_data, "license", None)
    license_name = _license(item, detected)
    revision = str(info.sha)
    mode = str(item.get("mode", "mixed"))
    category, quota_group = _defaults(item, "huggingface")
    files = [path for path in api.list_repo_files(repo, repo_type=repo_type, revision=revision) if _include_path(path, mode)]
    for path in files[: int(item.get("max_files", 10_000))]:
        local = Path(hf_hub_download(repo, path, repo_type=repo_type, revision=revision, token=os.getenv("HF_TOKEN")))
        if local.stat().st_size > max_file_bytes:
            continue
        try:
            text = local.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        chunks = [normalize_text(text)] if mode == "structured" else _chunks(text, int(item.get("max_chars", 16_000)))
        for chunk_index, chunk in enumerate(chunks):
            yield {
                "text": chunk,
                "source": f"hf_{repo_type}",
                "repo": repo,
                "doc_id": _doc_id("hf", repo, revision, path, chunk_index),
                "license": license_name,
                "url": f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo}/blob/{revision}/{path}",
                "path": path,
                "revision": revision,
                "category": category,
                "quota_group": quota_group,
            }


def collect_web(item: dict[str, Any], client: HttpClient) -> Iterator[dict[str, Any]]:
    url = str(item["url"])
    license_name = _license(item)
    revision = str(item["revision"])
    parsed = urlparse(url)
    robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
    robots = RobotFileParser()
    robots.set_url(robots_url)
    robots_response = client.session.get(robots_url, timeout=30)
    robots.parse(robots_response.text.splitlines() if robots_response.status_code < 400 else [])
    user_agent = str(client.session.headers["User-Agent"])
    if not robots.can_fetch(user_agent, url):
        raise PermissionError(f"robots.txt disallows collection: {url}")
    response = client.get(url)
    parser = MainTextParser()
    parser.feed(response.text)
    category, quota_group = _defaults(item, "web")
    for chunk_index, chunk in enumerate(_chunks(parser.text(), int(item.get("max_chars", 16_000)))):
        yield {
            "text": chunk,
            "source": "web_allowlist",
            "repo": parsed.netloc,
            "doc_id": _doc_id("web", parsed.netloc, revision, parsed.path, chunk_index),
            "license": license_name,
            "url": url,
            "path": parsed.path,
            "revision": revision,
            "category": category,
            "quota_group": quota_group,
        }


def collect_item(
    item: dict[str, Any],
    client: HttpClient,
    max_file_bytes: int,
    max_archive_bytes: int,
) -> Iterable[dict[str, Any]]:
    provider = str(item["provider"])
    if provider == "github":
        return collect_github(item, client, max_file_bytes, max_archive_bytes)
    if provider == "huggingface":
        return collect_huggingface(item, max_file_bytes)
    if provider == "web":
        return collect_web(item, client)
    raise ValueError(f"unsupported provider: {provider}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-token-env", default="GITHUB_TOKEN")
    parser.add_argument("--user-agent", default="CC-LLM-Research/1.0 (contact required in production)")
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--max-file-bytes", type=int, default=1_000_000)
    parser.add_argument("--max-archive-bytes", type=int, default=500_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = HttpClient(os.getenv(args.github_token_env), args.user_agent, args.delay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures_path = args.output.with_name(f"{args.output.stem}.failures.jsonl")
    records = 0
    skipped = 0
    with (
        args.output.open("wt", encoding="utf-8", newline="\n") as output,
        failures_path.open("wt", encoding="utf-8", newline="\n") as failures,
    ):
        for item in iter_jsonl([args.manifest]):
            try:
                for row in collect_item(item, client, args.max_file_bytes, args.max_archive_bytes):
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    records += 1
            except ArchiveTooLargeError as error:
                failure = {
                    "source": item,
                    "error": str(error),
                    "error_type": type(error).__name__,
                }
                failures.write(json.dumps(failure, ensure_ascii=False, separators=(",", ":")) + "\n")
                skipped += 1

    if skipped == 0:
        failures_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": records,
                "skipped": skipped,
                "failures": str(failures_path) if skipped else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

if __name__ == "__main__":
    main()

