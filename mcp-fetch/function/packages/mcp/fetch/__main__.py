"""MCP Fetch — DO Function edition.

Fetches a URL and returns its content as markdown (or raw HTML).
Designed to be called by agents as a lightweight tool function.

POST body:
  {"url": "https://example.com", "max_length": 5000, "start_index": 0, "raw": false}

Response:
  {"content": "# Example ...", "url": "https://example.com", "mime_type": "text/html"}
"""

import os
import re

import requests
from bs4 import BeautifulSoup

USER_AGENT = os.environ.get("USER_AGENT", "MCP-Fetch/1.0 (DO Functions)")
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "5000"))


def _html_to_markdown(html: str) -> str:
    """Simple HTML to markdown conversion using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav, footer
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Try to find main content
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    text = main.get_text(separator="\n", strip=True)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main(args):
    url = args.get("url", "")
    if not url:
        return {
            "body": {"error": "Missing required 'url' parameter"},
            "statusCode": 400,
        }

    max_length = int(args.get("max_length", DEFAULT_MAX_LENGTH))
    start_index = int(args.get("start_index", 0))
    raw = args.get("raw", False)

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=25,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return {
            "body": {"error": f"Fetch failed: {str(e)}", "url": url},
            "statusCode": 502,
        }

    content_type = resp.headers.get("content-type", "")
    mime_type = content_type.split(";")[0].strip()

    if raw or "html" not in mime_type:
        content = resp.text
    else:
        content = _html_to_markdown(resp.text)

    # Paginate
    total_length = len(content)
    content = content[start_index : start_index + max_length]

    result = {
        "content": content,
        "url": str(resp.url),
        "mime_type": mime_type,
        "total_length": total_length,
    }

    if start_index + max_length < total_length:
        result["next_start_index"] = start_index + max_length
        result["truncated"] = True

    return {"body": result}
