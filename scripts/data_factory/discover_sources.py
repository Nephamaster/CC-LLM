"""Discover candidate repositories and web pages for Phase 1 collection."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests

from scripts.data_factory.io_utils import iter_jsonl
from scripts.data_factory.text import VERIFIED_LICENSES, canonical_license


DEFAULT_SEARCHES = (
    "中文 人工智能",
    "中文 机器学习",
    "中文 自然语言处理",
    "中文 深度学习",
    "中文 大语言模型",
    "中文 数据科学",
    "中文 软件开发",
    "Chinese documentation",
)

# DEFAULT_SEARCHES = (
#     "人工智能",
#     "机器学习",
#     "自然语言处理",
#     "深度学习",
#     "大语言模型",
#     "数据科学",
#     "软件开发",
#     "智能体",
#     "低代码",
#     "检索",
#     "训练"
#     "推理",
#     "数据库",
#     "大数据",
#     "数据集",
#     "评估",
#     "微调",
#     "Chinese doc",
#     "vLLM",
#     "modelscope",
#     "BAAI",
#     "Qwen",
#     "kimi",
#     "Minimax",
#     "GLM",
#     "LangChain",
#     "LangGraph",
#     "Milvus",
#     "swift",
#     "transformers",
#     "datasets",
#     "MMLU",
#     "ARC",
#     "SQuAD",
#     "LoRA",
#     "SFT",
#     "DPO",
#     "RLHF"
# )

DEFAULT_LICENSES = (
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MPL-2.0",
    "CC-BY-4.0",
)


def _allowed_licenses(values: Iterable[str]) -> set[str]:
    allowed = {canonical_license(value) for value in values}
    unknown = sorted(allowed - VERIFIED_LICENSES)
    if unknown:
        raise ValueError(f"unsupported licenses: {', '.join(unknown)}")
    return allowed


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _github_candidates(args: argparse.Namespace, allowed: set[str]) -> list[dict[str, Any]]:
    token = os.getenv(args.github_token_env)
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": args.user_agent,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    candidates: dict[str, dict[str, Any]] = {}
    with requests.Session() as session:
        session.headers.update(headers)
        for search in args.search:
            remaining = args.limit_per_query
            page = 1
            while remaining > 0:
                per_page = min(remaining, 100)
                query = f'{search} in:readme stars:>={args.min_stars} archived:false fork:false'
                response = session.get(
                    "https://api.github.com/search/repositories",
                    params={"q": query, "sort": "stars", "per_page": per_page, "page": page},
                    timeout=args.timeout,
                )
                response.raise_for_status()
                items = response.json().get("items", [])
                if not items:
                    break
                for item in items:
                    license_info = item.get("license") or {}
                    license_name = canonical_license(license_info.get("spdx_id", ""))
                    repo = item["full_name"]
                    if license_name not in allowed or repo in candidates:
                        continue
                    candidates[repo] = {
                        "provider": "github",
                        "repo": repo,
                        "license": license_name,
                        "mode": args.mode,
                        "discovered_by": "github_search",
                        "search": search,
                        "quality": {
                            "stars": item.get("stargazers_count", 0),
                            "forks": item.get("forks_count", 0),
                        },
                    }
                remaining -= len(items)
                if len(items) < per_page:
                    break
                page += 1
                time.sleep(args.delay)
    return list(candidates.values())


def _card_value(card_data: Any, key: str) -> Any:
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        return card_data.get(key)
    return getattr(card_data, key, None)


def _hf_license(info: Any) -> str:
    declared = _card_value(getattr(info, "card_data", None), "license")
    values = declared if isinstance(declared, list) else [declared]
    values.extend(
        tag.removeprefix("license:")
        for tag in (getattr(info, "tags", None) or [])
        if tag.startswith("license:")
    )
    for value in values:
        if value:
            normalized = canonical_license(str(value))
            if normalized in VERIFIED_LICENSES:
                return normalized
    return ""


def _hf_candidates(args: argparse.Namespace, allowed: set[str]) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for Hugging Face discovery") from exc

    api = HfApi(token=os.getenv(args.hf_token_env))
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for search in args.search:
        for repo_type in args.hf_repo_types:
            method = api.list_models if repo_type == "model" else api.list_datasets
            infos = method(search=search, limit=args.limit_per_query, full=True)
            for info in infos:
                license_name = _hf_license(info)
                repo = getattr(info, "id", "")
                key = (repo_type, repo)
                if not repo or license_name not in allowed or key in candidates:
                    continue
                row: dict[str, Any] = {
                    "provider": "huggingface",
                    "repo": repo,
                    "repo_type": repo_type,
                    "license": license_name,
                    "mode": args.mode,
                    "discovered_by": "huggingface_search",
                    "search": search,
                    "quality": {
                        "downloads": getattr(info, "downloads", 0) or 0,
                        "likes": getattr(info, "likes", 0) or 0,
                    },
                }
                if getattr(info, "sha", None):
                    row["revision"] = info.sha
                candidates[key] = row
    return list(candidates.values())


def discover_repositories(args: argparse.Namespace) -> None:
    allowed = _allowed_licenses(args.licenses)
    rows: list[dict[str, Any]] = []
    if "github" in args.provider:
        rows.extend(_github_candidates(args, allowed))
    if "huggingface" in args.provider:
        rows.extend(_hf_candidates(args, allowed))
    rows.sort(
        key=lambda row: (
            row["provider"],
            -sum(row.get("quality", {}).values()),
            row["repo"],
        )
    )
    count = _write_jsonl(args.output, rows)
    providers = Counter(row["provider"] for row in rows)
    print(json.dumps({"output": str(args.output), "count": count, "providers": providers}, ensure_ascii=False))


def _sitemap_locations(content: bytes) -> tuple[bool, list[str]]:
    if content.startswith(b"\x1f\x8b"):
        content = gzip.decompress(content)
    root = ET.fromstring(content)
    is_index = root.tag.rsplit("}", 1)[-1] == "sitemapindex"
    locations = [
        (element.text or "").strip()
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "loc" and (element.text or "").strip()
    ]
    return is_index, locations


def _crawl_sitemap(
    site: dict[str, Any], args: argparse.Namespace, session: requests.Session
) -> list[dict[str, Any]]:
    sitemap = site["sitemap"]
    license_name = canonical_license(site["license"])
    if license_name not in VERIFIED_LICENSES:
        raise ValueError(f"unsupported license for {sitemap}: {license_name}")
    if not site.get("revision"):
        raise ValueError(f"revision is required for sitemap source: {sitemap}")

    host = urlparse(sitemap).netloc.lower()
    include = re.compile(site["include"]) if site.get("include") else None
    exclude = re.compile(site["exclude"]) if site.get("exclude") else None
    max_urls = int(site.get("max_urls", args.max_urls_per_site))
    queue = deque([sitemap])
    visited: set[str] = set()
    urls: list[str] = []

    while queue and len(visited) < args.max_sitemaps and len(urls) < max_urls:
        current = queue.popleft()
        if current in visited or urlparse(current).netloc.lower() != host:
            continue
        visited.add(current)
        response = session.get(current, timeout=args.timeout)
        response.raise_for_status()
        is_index, locations = _sitemap_locations(response.content)
        if is_index:
            queue.extend(locations)
        else:
            for url in locations:
                if urlparse(url).netloc.lower() != host:
                    continue
                if include and not include.search(url):
                    continue
                if exclude and exclude.search(url):
                    continue
                urls.append(url)
                if len(urls) >= max_urls:
                    break
        time.sleep(args.delay)

    return [
        {
            "provider": "web",
            "url": url,
            "license": license_name,
            "revision": site["revision"],
            "mode": site.get("mode", args.mode),
            "discovered_by": sitemap,
        }
        for url in dict.fromkeys(urls)
    ]


def discover_sitemaps(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": args.user_agent})
        for site in iter_jsonl([args.sites]):
            rows.extend(_crawl_sitemap(site, args, session))
    count = _write_jsonl(args.output, rows)
    print(json.dumps({"output": str(args.output), "count": count}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    repositories = subparsers.add_parser("repositories", help="discover GitHub and Hugging Face repositories")
    repositories.add_argument("--output", type=Path, required=True)
    repositories.add_argument(
        "--provider",
        nargs="+",
        choices=("github", "huggingface"),
        default=["github", "huggingface"],
    )
    repositories.add_argument("--search", action="append", default=None, help="repeatable search expression")
    repositories.add_argument("--mode", choices=("mixed", "code", "structured"), default="mixed")
    repositories.add_argument("--limit-per-query", type=int, default=100)
    repositories.add_argument("--min-stars", type=int, default=10)
    repositories.add_argument("--licenses", nargs="+", default=list(DEFAULT_LICENSES))
    repositories.add_argument("--hf-repo-types", nargs="+", choices=("model", "dataset"), default=["model", "dataset"])
    repositories.add_argument("--github-token-env", default="GITHUB_TOKEN")
    repositories.add_argument("--hf-token-env", default="HF_TOKEN")
    repositories.add_argument("--user-agent", default="CC-LLM-DataFactory/1.0")
    repositories.add_argument("--timeout", type=float, default=30.0)
    repositories.add_argument("--delay", type=float, default=0.2)
    repositories.set_defaults(func=discover_repositories)

    sitemap = subparsers.add_parser("sitemap", help="expand approved sitemap sources into web manifest entries")
    sitemap.add_argument("--sites", type=Path, required=True)
    sitemap.add_argument("--output", type=Path, required=True)
    sitemap.add_argument("--mode", choices=("mixed", "code", "structured"), default="mixed")
    sitemap.add_argument("--max-urls-per-site", type=int, default=10_000)
    sitemap.add_argument("--max-sitemaps", type=int, default=100)
    sitemap.add_argument("--user-agent", default="CC-LLM-DataFactory/1.0")
    sitemap.add_argument("--timeout", type=float, default=30.0)
    sitemap.add_argument("--delay", type=float, default=0.2)
    sitemap.set_defaults(func=discover_sitemaps)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "search", None) is None:
        args.search = list(DEFAULT_SEARCHES)
    args.func(args)


if __name__ == "__main__":
    main()
