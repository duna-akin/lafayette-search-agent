import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Annotated, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os
import getpass
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
import urllib.parse
import time
import re
from urllib.parse import urlparse, parse_qs
from functools import lru_cache
from datetime import datetime, timedelta

# -----------------------
# Bootstrap / Config
# -----------------------
load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# You can swap the model as you like
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

# High-signal stats pages — always try these first
STATS_PAGES = [
    "https://about.lafayette.edu/lafayette-at-a-glance/",
    "https://oir.lafayette.edu/",
    "https://oir.lafayette.edu/common-data-set/",
]

# Topic maps (kept, but augmented)
LAFAYETTE_URL_MAP = {
    "admissions": [
        "https://admissions.lafayette.edu/",
        "https://admissions.lafayette.edu/what-we-look-for/",
        "https://admissions.lafayette.edu/apply/",
        "https://admissions.lafayette.edu/deadlines-and-forms/",
        "https://admissions.lafayette.edu/admissions-visits/",
    ],
    "financial": [
        "https://admissions.lafayette.edu/financial-aid/",
        "https://admissions.lafayette.edu/college-costs/",
    ],
    "academics": [
        "https://academics.lafayette.edu/",
        "https://academics.lafayette.edu/departments-programs/",
    ],
    "campus": [
        "https://campuslife.lafayette.edu/",
    ],
    "about": [
        "https://about.lafayette.edu/",
        "https://about.lafayette.edu/lafayette-at-a-glance/",
        "https://about.lafayette.edu/mission-and-history/",
        "https://about.lafayette.edu/why-not/",
    ],
    "president": [
        "https://president.lafayette.edu/",
    ],
}

# -----------------------
# Utilities
# -----------------------
def extract_real_urls_from_google(soup: BeautifulSoup) -> List[str]:
    """
    Unwrap Google SERP links like /url?q=<target> and return lafayette.edu URLs.
    """
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Google redirect
        if href.startswith("/url?"):
            q = parse_qs(urlparse(href).query).get("q", [""])[0]
            if q and "lafayette.edu" in q:
                out.append(q)
        # Rare direct links
        elif href.startswith("http") and "lafayette.edu" in href:
            out.append(href)
    # Dedup, preserve order
    seen = set()
    cleaned = []
    for u in out:
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


def extract_stats_blob(text: str) -> Dict[str, str]:
    """
    Very simple stat sniffer that catches common phrasing.
    Extend patterns as needed.
    """
    stats = {}

    # Undergrad enrollment / total students
    m = re.search(r"\b(\d{3,5})\b\s+(?:undergraduates|undergraduate students|students)\b", text, re.I)
    if m:
        stats["undergrad_enrollment_guess"] = m.group(1)

    # Student-faculty ratio like 10:1 or 9 to 1
    m = re.search(
        r"(\d{1,2}\s*[:to]\s*\d{1,2})\s*(?:student[- ]to[- ]faculty|student[- ]faculty|faculty[- ]student)\s*ratio",
        text,
        re.I,
    )
    if m:
        # normalize "9 to 1" -> "9:1"
        ratio = m.group(1).replace(" ", "")
        ratio = ratio.replace("to", ":")
        stats["student_faculty_ratio_guess"] = ratio

    # Average class size
    m = re.search(r"(?:average|avg)\s+class\s+size\s*(\d{1,2})", text, re.I)
    if m:
        stats["avg_class_size_guess"] = m.group(1)

    # Grad enrollment (rare, but include)
    m = re.search(r"\b(\d{2,5})\b\s+(?:graduate students|graduates)\b", text, re.I)
    if m:
        stats["grad_enrollment_guess"] = m.group(1)

    return stats


# naive in-memory cache with TTL
_SCRAPE_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = timedelta(hours=24)


def get_cached(url: str):
    entry = _SCRAPE_CACHE.get(url)
    if not entry:
        return None
    if datetime.utcnow() - entry["ts"] > CACHE_TTL:
        _SCRAPE_CACHE.pop(url, None)
        return None
    return entry["data"]


def put_cached(url: str, data: Dict[str, Any]):
    _SCRAPE_CACHE[url] = {"data": data, "ts": datetime.utcnow()}


def scrape_url(url: str) -> Dict[str, Any]:
    """
    Scrape a single URL. Keep header/aside (stats often live there).
    Return text (truncated), and stats extracted.
    """
    cached = get_cached(url)
    if cached:
        return cached

    try:
        time.sleep(0.6)  # be polite
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        # Remove noisy tags but KEEP header/aside
        for tag in ["script", "style", "nav", "footer"]:
            for el in soup.find_all(tag):
                el.decompose()

        # Try to scope to main-ish content; fall back to body
        main = None
        for selector in ["main", ".main-content", ".content", "#content", "article", ".page-content"]:
            main = soup.select_one(selector)
            if main:
                break
        if not main:
            main = soup.find("body") or soup

        text = main.get_text(" ", strip=True)
        # More generous cap (many pages are long)
        if len(text) > 25000:
            text = text[:25000]

        stats = extract_stats_blob(text)
        data = {"url": url, "text": text, "stats": stats}
        put_cached(url, data)
        return data

    except Exception as e:
        return {"url": url, "text": "", "stats": {}, "error": str(e)}


def google_site_search(query: str, limit: int = 4) -> List[str]:
    """
    Google site:lafayette.edu query, unwrap links, return up to `limit` URLs.
    """
    try:
        search_query = f"site:lafayette.edu {query}"
        google_search_url = f"https://www.google.com/search?q={urllib.parse.quote(search_query)}"
        resp = requests.get(google_search_url, headers=REQUEST_HEADERS, timeout=12)
        soup = BeautifulSoup(resp.content, "html.parser")
        urls = extract_real_urls_from_google(soup)
        return urls[:limit]
    except Exception:
        return []


def route_urls_by_query(query: str) -> List[str]:
    """
    Build a prioritized URL list from:
    1) Stats pages (always)
    2) Keyword routing maps
    3) Google site: search
    4) Fallback core pages
    """
    q = query.lower().strip()
    prioritized = list(STATS_PAGES)  # seed with stats pages

    # Expanded routing for stats-ish queries
    if any(w in q for w in [
        "enrollment", "number of students", "student body", "headcount",
        "ratio", "student-faculty", "faculty student", "class size", "average class",
        "acceptance rate", "common data set", "cds"
    ]):
        prioritized.extend(STATS_PAGES)

    if any(w in q for w in ['admission', 'apply', 'application', 'deadline', 'requirement', 'ed', 'early decision', 'regular decision']):
        prioritized.extend(LAFAYETTE_URL_MAP['admissions'])
    if any(w in q for w in ['financial', 'aid', 'scholarship', 'cost', 'tuition', 'money', 'grant']):
        prioritized.extend(LAFAYETTE_URL_MAP['financial'])
    if any(w in q for w in ['academic', 'major', 'program', 'course', 'department', 'study', 'curriculum']):
        prioritized.extend(LAFAYETTE_URL_MAP['academics'])
    if any(w in q for w in ['campus', 'life', 'student', 'housing', 'dining', 'club', 'organization']):
        prioritized.extend(LAFAYETTE_URL_MAP['campus'])
    if any(w in q for w in ['about', 'history', 'mission', 'overview', 'glance', 'why']):
        prioritized.extend(LAFAYETTE_URL_MAP['about'])
    if 'president' in q:
        prioritized.extend(LAFAYETTE_URL_MAP['president'])

    # Google site search (adds diversity / deep pages)
    google_urls = google_site_search(query, limit=4)
    prioritized.extend(google_urls)

    # Fallbacks if still too thin
    if len(prioritized) < 3:
        prioritized.extend([
            "https://admissions.lafayette.edu/",
            "https://academics.lafayette.edu/",
            "https://about.lafayette.edu/",
        ])

    # Dedup while preserving order
    seen = set()
    unique_urls = []
    for u in prioritized:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    return unique_urls


def search_lafayette_edu(query: str, max_pages: int = 7) -> Dict[str, Any]:
    """
    Orchestrates URL selection + scraping.
    Returns combined text and merged stats from all pages.
    """
    urls = route_urls_by_query(query)
    scraped = []
    for url in urls[:max_pages]:
        # Removed scraping progress messages from UI - only print to console for debugging
        # print(f"Scraping: {url}")  # Uncomment this line if you want console logging
        data = scrape_url(url)
        if data.get("text"):
            scraped.append(data)

    if not scraped:
        fallback = (
            "Lafayette College is a private liberal arts college in Easton, Pennsylvania. "
            "For specific statistics (enrollment, student-faculty ratio, etc.), see "
            "the At-a-Glance page or the Office of Institutional Research / Common Data Set."
        )
        return {"combined_text": fallback, "stats": {}, "sources": []}

    combined_text = "\n\n".join([f"From {d['url']}:\n{d['text']}" for d in scraped])

    # Merge stats (last-writer-wins is fine; you can prioritize by page later)
    merged_stats: Dict[str, str] = {}
    for d in scraped:
        for k, v in d.get("stats", {}).items():
            if v and k not in merged_stats:
                merged_stats[k] = v

    sources = [d["url"] for d in scraped]
    return {"combined_text": combined_text, "stats": merged_stats, "sources": sources}


# -----------------------
# LangGraph Wiring
# -----------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def chatbot(state: State):
    """Lafayette College Information Assistant — reliable stats-first answers."""
    last_message = state["messages"][-1]

    if isinstance(last_message, dict):
        user_question = last_message.get("content", "")
    else:
        # LangChain message object
        user_question = getattr(last_message, "content", "")


    # Scrape & aggregate
    search_payload = search_lafayette_edu(user_question)
    combined_text = search_payload["combined_text"]
    stats = search_payload["stats"]
    sources = search_payload["sources"]

    # Build a precise, helpful system prompt
    SYSTEM_PROMPT = f"""
You are a Lafayette College information assistant. Answer ONLY about Lafayette College.

SCRAPED_TEXT:
{combined_text}

STATS_EXTRACTED (quick regex guesses; prefer these when present):
{stats}

SOURCES (pages scraped for this answer):
{sources}

USER QUESTION:
{user_question}

INSTRUCTIONS:
- If STATS_EXTRACTED contains the relevant number (e.g., undergrad_enrollment_guess, student_faculty_ratio_guess, avg_class_size_guess), use it and cite which SOURCE the data likely came from (by URL).
- If not in STATS_EXTRACTED, search SCRAPED_TEXT for specifics; quote exact figures when clear and cite the SOURCE URL.
- If still unknown, say exactly which SOURCE URLs you checked and suggest the precise next Lafayette page to try (e.g., At-a-Glance or OIR Common Data Set).
- Do NOT invent numbers. Be concise and clear. Provide the number first, then a one-line cite.
"""

    formatted_response = llm.invoke([HumanMessage(content=SYSTEM_PROMPT)])
    response_message = AIMessage(content=formatted_response.content)
    return {"messages": state["messages"] + [response_message]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(
    page_title="Lafayette College Information Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Lafayette College Information Assistant")
st.markdown("### Your AI guide to everything Lafayette College!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("What would you like to know about Lafayette College?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Lafayette College website..."):
            try:
                result = graph.invoke({"messages": [HumanMessage(content=prompt)]})
                response = result["messages"][-1].content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error while searching Lafayette College information: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar
with st.sidebar:
    st.header("🎓 About Lafayette College Assistant")
    st.markdown("""
This AI assistant pulls directly from Lafayette's official site, prioritizing **At-a-Glance** and **Office of Institutional Research** pages for accurate stats.
    
**I can help with:**
- Admissions (deadlines, requirements)
- Academics (majors, departments)
- Campus life (housing, dining, clubs)
- Financial aid (scholarships, costs)
- Quick stats (enrollment, student–faculty ratio, class size)

**Quick Links:**
- [Lafayette At-a-Glance](https://about.lafayette.edu/lafayette-at-a-glance/)
- [Office of Institutional Research](https://oir.lafayette.edu/)
- [Common Data Set](https://oir.lafayette.edu/common-data-set/)
- [Admissions](https://admissions.lafayette.edu/)
- [Academics](https://academics.lafayette.edu/)
""")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
