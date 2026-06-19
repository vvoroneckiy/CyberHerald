import time
import re
import html2text
import feedparser
from duckduckgo_search import DDGS
from typing import Optional


RSS_FEEDS = [
    "https://tass.ru/rss/v2.xml",
    "https://ria.ru/export/rss2/archive/index.xml",
    "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "https://iz.ru/export/rss2/archive/index.xml",
    "https://lenta.ru/rss/",
]

_html_converter = html2text.HTML2Text()
_html_converter.ignore_links = True
_html_converter.ignore_images = True
_html_converter.ignore_emphasis = True
_html_converter.body_width = 0


def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = _html_converter.handle(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_voice_query(query: str) -> str:
    q = query.strip()
    q_lower = q.lower()
    prefixes = [
        "расскажи мне про", "расскажи мне о", "расскажи про", "расскажи о",
        "расскажи мне", "расскажи",
        "найди мне", "найди", "поищи", "поиск",
        "что нового в", "что нового", "что такое",
    ]
    filler_words = {"пожалуйста", "будьте", "добры", "спасибо", "очень", "можно",
                    "кампания", "кампанию", "кампании"}
    for p in prefixes:
        if q_lower.startswith(p):
            rest = q[len(p):].strip().lstrip(",;:-. ")
            if rest:
                words = re.sub(r"[^\w\s-]", "", rest).split()
                words = [w for w in words if w.lower() not in filler_words]
                if words:
                    return " ".join(words).lower()
    return re.sub(r"[^\w\s-]", "", q).lower()


class WebSearchService:
    def __init__(self):
        self.client = DDGS()
        self._rss_cache = None
        self._rss_cache_time = 0

    def fetch_rss(self) -> list[dict]:
        now = time.time()
        if self._rss_cache and now - self._rss_cache_time < 120:
            return self._rss_cache

        articles = []
        seen = set()
        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url, agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
                for entry in feed.entries[:10]:
                    title = (entry.get("title") or "").strip()
                    if not title or title.lower() in seen:
                        continue
                    seen.add(title.lower())
                    summary = entry.get("summary") or entry.get("description") or ""
                    content = _clean_html(summary)
                    source = getattr(entry, "source", None)
                    source_title = getattr(source, "title", None) if source else None
                    articles.append({
                        "title": title,
                        "content": content,
                        "source": source_title or feed.feed.get("title", url.split("/")[2]),
                    })
            except Exception as e:
                print(f"[WARN] RSS fetch failed for {url}: {e}")
        self._rss_cache = articles
        self._rss_cache_time = now
        return articles

    def search_news(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            results = list(self.client.news(query, max_results=max_results))
            articles = []
            seen = set()
            for r in results:
                title = (r.get("title") or "").strip()
                if not title or title.lower() in seen:
                    continue
                seen.add(title.lower())
                articles.append({
                    "title": title,
                    "content": _clean_html(r.get("body", "")),
                    "source": r.get("source", r.get("url", "Web")),
                })
            return articles
        except Exception as e:
            print(f"[WARN] Web search_news failed: {e}")
            return []

    def search_general(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            results = list(self.client.text(query, max_results=max_results))
            articles = []
            seen = set()
            for r in results:
                title = (r.get("title") or "").strip()
                if not title or title.lower() in seen:
                    continue
                seen.add(title.lower())
                articles.append({
                    "title": title,
                    "content": _clean_html(r.get("body", "")),
                    "source": r.get("href", "Web"),
                })
            return articles
        except Exception as e:
            print(f"[WARN] Web search_general failed: {e}")
            return []

    def search_multi(self, queries: list[str], max_per_query: int = 4) -> list[dict]:
        seen = set()
        results = []
        rss = self.fetch_rss()
        for a in rss:
            key = a["title"].lower()
            if key not in seen:
                seen.add(key)
                results.append(a)
        for i, q in enumerate(queries):
            if len(results) >= 15:
                break
            if i > 0:
                time.sleep(1.5)
            articles = self.search_general(q, max_results=max_per_query)
            for a in articles:
                key = a["title"].lower()
                if key not in seen:
                    seen.add(key)
                    results.append(a)
        return results[:15]

    def search_bing_news(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            import urllib.parse
            q = urllib.parse.quote(query)
            url = f"https://www.bing.com/news/search?q={q}&format=rss"
            feed = feedparser.parse(url, agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
            articles = []
            seen = set()
            for entry in feed.entries[:max_results]:
                title = (entry.get("title") or "").strip()
                if not title or title.lower() in seen:
                    continue
                seen.add(title.lower())
                summary = entry.get("summary") or ""
                content = _clean_html(summary)
                source = getattr(entry, "news_source", None) or entry.get("source", "")
                articles.append({
                    "title": title,
                    "content": content,
                    "source": source if isinstance(source, str) else "Bing News",
                    "url": entry.get("link", ""),
                })
            return articles
        except Exception as e:
            print(f"[WARN] Bing News RSS failed: {e}")
            return []

    def search_voice(self, query: str, max_results: int = 5) -> list[dict]:
        clean_query = clean_voice_query(query)
        print(f"[DEBUG] search_voice: raw='{query}' clean='{clean_query}'")

        # 1. Bing News RSS — реальные новости по любому запросу
        results = self.search_bing_news(clean_query, max_results=max_results)
        print(f"[DEBUG] search_voice: Bing News => {len(results)}")

        # 2. DuckDuckGo news — если Bing не дал
        if len(results) == 0:
            ddg = self.search_news(clean_query, max_results=max_results)
            print(f"[DEBUG] search_voice: DDG news => {len(ddg)}")
            for a in ddg:
                key = a["title"].lower()
                if key not in {r["title"].lower() for r in results}:
                    results.append(a)

        # 3. DuckDuckGo text — последняя надежда
        if len(results) == 0:
            ddg = self.search_general(clean_query, max_results=max_results)
            print(f"[DEBUG] search_voice: DDG text => {len(ddg)}")
            for a in ddg:
                key = a["title"].lower()
                if key not in {r["title"].lower() for r in results}:
                    results.append(a)

        # 4. Если всё пусто — RSS-ленты
        if not results:
            fallback = self.fetch_rss()[:max_results]
            print(f"[DEBUG] search_voice: fallback RSS ({len(fallback)})")
            return fallback
        return results[:max_results]
