"""Wikipedia category traversal and article fetching for the philately corpus."""

from __future__ import annotations

import wikipediaapi

USER_AGENT = (
    "PhilatelyAssistant/0.1 "
    "(https://github.com/kozzzak/philately-assistant; kozzzak@gmail.com)"
)

DEFAULT_CATEGORIES = [
    "Philately",
    "Postage stamps by country",
    "Compendium of postage stamp issuers",
]


def make_wiki(language: str = "en") -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(user_agent=USER_AGENT, language=language)


def collect_article_titles(
    wiki: wikipediaapi.Wikipedia,
    categories: list[str],
    max_depth: int = 2,
    max_articles: int | None = None,
) -> set[str]:
    """BFS the category tree starting from `categories`, collecting article titles.

    Subcategories are followed up to `max_depth` levels deep. Stops early once
    `max_articles` distinct article titles have been collected.
    """
    seen_categories: set[str] = set()
    titles: set[str] = set()
    queue: list[tuple[str, int]] = [(f"Category:{c}", 0) for c in categories]

    while queue:
        cat_title, depth = queue.pop(0)
        if cat_title in seen_categories:
            continue
        seen_categories.add(cat_title)

        cat_page = wiki.page(cat_title)
        if not cat_page.exists():
            continue

        for member_title, member_page in cat_page.categorymembers.items():
            if member_page.ns == wikipediaapi.Namespace.CATEGORY:
                if depth < max_depth:
                    queue.append((member_title, depth + 1))
            elif member_page.ns == wikipediaapi.Namespace.MAIN:
                titles.add(member_title)
                if max_articles and len(titles) >= max_articles:
                    return titles

    return titles


def fetch_article(wiki: wikipediaapi.Wikipedia, title: str):
    """Fetch a single article page. Returns None if it doesn't exist."""
    page = wiki.page(title)
    if not page.exists():
        return None
    return page
