"""BestBuy.ca product-review scraper (Selenium + BeautifulSoup).

Design goals: public data only, polite, auditable, and it stops instead of fighting a block.
  * robots.txt is checked before any page is opened
  * randomised delays between actions + exponential-backoff retries for transient errors
  * a hard budget on 'Show More' clicks and on the number of reviews (so it can never run away)
  * any block / CAPTCHA page raises BlockedError and the run STOPS (no CAPTCHA solving, no stealth/evasion tricks)
"""
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser

import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (ElementClickInterceptedException, NoSuchElementException,
                                        StaleElementReferenceException, TimeoutException, WebDriverException)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

os.environ.setdefault("SE_AVOID_STATS", "true")          # Selenium Manager: don't send usage statistics

# CSS hooks discovered by inspecting the public reviews page (data-automation attributes are the most stable hooks the site offers)
SEL_ITEM = 'li[data-automation="review-list-item"]'
SEL_SHOW_MORE = 'button[data-automation="load-more"]'
SEL_SORT = "select#Sort"
SEL_VERIFIED = "#verifiedPurchaserToggle"
BLOCK_MARKERS = ("access denied", "captcha", "unusual traffic", "pardon our interruption", "request unsuccessful",
                 "are you a robot", "verify you are human", "temporarily blocked", "incident id")


class BlockedError(RuntimeError):
    """Raised when the site shows a block / bot-check page. The scraper stops rather than trying to get around it."""


@dataclass
class ScrapeConfig:
    product_url: str                                    # canonical product URL (no tracking parameters)
    sku: str
    product_name: str = ""
    sorts: tuple = ("Newest", "Most Helpful", "Most Relevant", "Highest Rating", "Lowest Rating", "Oldest")
    full_sort: str = "Newest"                           # this sort is paged until 'Show More' disappears -> collects EVERY review
    shallow_clicks: int = 1                             # other sorts: only this many 'Show More' clicks (first ~20 reviews, used for rankings)
    verified_pass: bool = True                          # repeat full_sort with 'Verified Buyers Only' to flag verified purchases
    max_reviews: int = 200                              # hard ceiling on unique reviews kept
    max_total_clicks: int = 60                          # hard ceiling on 'Show More' clicks for the whole run
    delay_range: tuple = (2.5, 5.0)                     # seconds, randomised between actions
    headless: bool = False                              # a normal (visible) Chrome window looks like an ordinary visitor
    proxy: str = None                                   # e.g. "http://user:pass@host:port" - supported but NOT used in this run
    page_timeout: int = 40
    out_dir: str = "outputs"
    user_agent_note: str = "default Chrome user-agent (not spoofed)"


# --------------------------------------------------------------------------------------------------------------------------
# Politeness + resilience helpers
# --------------------------------------------------------------------------------------------------------------------------
def robots_allows(url, user_agent="*"):
    """True if robots.txt permits fetching `url`. (robots.txt is fetched with `requests` so a normal UA is sent.)"""
    root = "/".join(url.split("/")[:3])
    txt = requests.get(root + "/robots.txt", timeout=20, headers={"User-Agent": "Mozilla/5.0"}).text
    rp = robotparser.RobotFileParser()
    rp.parse(txt.splitlines())
    return rp.can_fetch(user_agent, url), txt


def polite_sleep(cfg, extra=0.0):
    """Randomised pause - fixed intervals look robotic and a human never clicks at exact 2.0 s intervals."""
    time.sleep(random.uniform(*cfg.delay_range) + extra)


def with_retries(fn, attempts=3, base_delay=3.0, retry_on=(StaleElementReferenceException, TimeoutException, ElementClickInterceptedException, WebDriverException), log=None):
    """Run fn(); on a transient error wait base_delay * 2^k (+ jitter) and retry. BlockedError is never retried."""
    for k in range(attempts):
        try:
            return fn()
        except BlockedError:
            raise
        except retry_on as e:
            if k == attempts - 1:
                raise
            wait = base_delay * (2 ** k) + random.uniform(0, 1.5)
            if log is not None:
                log.append({"event": "retry", "error": type(e).__name__, "wait_s": round(wait, 1)})
            time.sleep(wait)


def check_not_blocked(driver):
    head = (driver.title or "").lower() + " " + driver.page_source[:4000].lower()
    hit = next((m for m in BLOCK_MARKERS if m in head), None)
    if hit:
        raise BlockedError(f"Block / bot-check page detected (matched '{hit}'). Stopping as agreed - no workaround attempted.")


def make_driver(cfg):
    o = Options()
    if cfg.headless:
        o.add_argument("--headless=new")
    o.add_argument("--window-size=1366,1000")
    o.add_argument("--lang=en-CA")
    if cfg.proxy:
        o.add_argument(f"--proxy-server={cfg.proxy}")
    d = webdriver.Chrome(options=o)
    d.set_page_load_timeout(cfg.page_timeout)
    return d


# --------------------------------------------------------------------------------------------------------------------------
# Page interaction
# --------------------------------------------------------------------------------------------------------------------------
def dismiss_cookie_banner(driver):
    """Choose the most privacy-preserving option if a consent banner appears (decline / reject / necessary only)."""
    for b in driver.find_elements(By.CSS_SELECTOR, "button"):
        try:
            if re.search(r"decline|reject|necessary only|refuse", b.text or "", re.I) and b.is_displayed():
                driver.execute_script("arguments[0].click()", b)
                return True
        except StaleElementReferenceException:
            continue
    return False


def open_reviews_page(driver, cfg, log):
    url = cfg.product_url.rstrip("/") + "/review"
    with_retries(lambda: driver.get(url), log=log)
    check_not_blocked(driver)
    WebDriverWait(driver, 25).until(EC.presence_of_element_located((By.CSS_SELECTOR, SEL_ITEM)))
    dismiss_cookie_banner(driver)
    log.append({"event": "opened", "url": url})


def apply_sort(driver, cfg, label, log):
    """Pick a sort order in the Sort drop-down and wait for the list to be re-rendered."""
    first = driver.find_element(By.CSS_SELECTOR, SEL_ITEM)
    sel = Select(driver.find_element(By.CSS_SELECTOR, SEL_SORT))
    option = next(o for o in sel.options if o.text.strip().lower() == label.lower())
    sel.select_by_visible_text(option.text)
    try:
        WebDriverWait(driver, 10).until(EC.staleness_of(first))        # the old first row disappears when the list reloads
    except TimeoutException:
        pass                                                           # same top review under both sorts is possible
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, SEL_ITEM)))
    polite_sleep(cfg)
    check_not_blocked(driver)
    log.append({"event": "sort", "label": label})


def set_verified_only(driver, cfg, on, log):
    """Toggle the 'Verified Buyers Only' filter (a role=switch checkbox)."""
    box = driver.find_element(By.CSS_SELECTOR, SEL_VERIFIED)
    if box.is_selected() == on:
        return
    first = driver.find_element(By.CSS_SELECTOR, SEL_ITEM)
    driver.execute_script("arguments[0].scrollIntoView({block:'center'}); arguments[0].click();", box)
    try:
        WebDriverWait(driver, 10).until(EC.staleness_of(first))
    except TimeoutException:
        pass
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, SEL_ITEM)))
    polite_sleep(cfg)
    log.append({"event": "verified_only", "on": on})


def click_show_more(driver):
    """Click 'Show More' if it exists. Returns True if clicked, False when there is nothing more to load."""
    btns = driver.find_elements(By.CSS_SELECTOR, SEL_SHOW_MORE)
    if not btns or not btns[0].is_displayed() or not btns[0].is_enabled():
        return False
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btns[0])
    time.sleep(0.6)
    driver.execute_script("arguments[0].click();", btns[0])
    return True


# --------------------------------------------------------------------------------------------------------------------------
# Parsing (BeautifulSoup on each review's outerHTML)
# --------------------------------------------------------------------------------------------------------------------------
def make_review_id(sku, reviewer, date, title, body):
    """Stable primary key. The page exposes no review id, so one is derived from the review's own content
    (the same review always gets the same id, and two different reviews practically never collide)."""
    key = f"{reviewer}|{date}|{title}|{body[:200]}"
    return f"BB{sku}-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10].upper()


def parse_review(html, sku):
    """Turn one <li> review into a dict of the required fields (+ a few extras)."""
    li = BeautifulSoup(html, "lxml")
    txt = lambda el: re.sub(r"\s+", " ", el.get_text(" ", strip=True)) if el else ""

    m = re.search(r"(\d)\s*out of\s*5", txt(li.select_one("p.sr-only")))
    rating = int(m.group(1)) if m else None

    title = txt(li.select_one("h3"))
    reviewer = re.sub(r"^\s*Reviewed by\s*", "", txt(li.select_one("[class*='author_']")), flags=re.I)

    date_el = li.select_one('[data-automation="rating-date"]')
    date = None
    if date_el is not None and date_el.get("data-date"):
        date = date_el["data-date"][:10]                                 # ISO timestamp -> YYYY-MM-DD
    elif date_el is not None:
        try:
            date = datetime.strptime(txt(date_el).lstrip("- ").strip(), "%B %d, %Y").strftime("%Y-%m-%d")
        except ValueError:
            pass

    recommended = None
    rec_el = li.select_one('[data-automation="friend-recommendation-text-container"]')
    if rec_el is not None:
        rtxt = txt(rec_el).lower()
        recommended = True if "yes" in rtxt[:6] else False if "no" in rtxt[:6] else None
        rec_el.extract()                                                 # keep the recommendation line out of the review text
    body = txt(li.select_one("[class*='reviewContent']"))

    helpful = not_helpful = None
    for b in li.find_all("button"):
        t = txt(b)
        mh = re.match(r"Helpful \((\d+)\)", t)
        mn = re.match(r"Not helpful \((\d+)\)", t)
        helpful = int(mh.group(1)) if mh else helpful
        not_helpful = int(mn.group(1)) if mn else not_helpful

    # sub-ratings shown on most reviews: "Quality: | Excellent | Value: | Good | Ease of Use: | Excellent"
    flat = li.get_text(" | ", strip=True)
    sub = lambda name: (re.search(name + r":\s*\|?\s*([A-Za-z ]+?)\s*\|", flat) or [None, None])[1]

    return {"review_id": make_review_id(sku, reviewer, date, title, body),
            "title": title, "review_text": body, "date": date, "rating": rating, "reviewer_name": reviewer,
            "helpful_votes": helpful, "not_helpful_votes": not_helpful, "recommended": recommended,
            "quality": sub("Quality"), "value": sub("Value"), "ease_of_use": sub("Ease of Use")}


# --------------------------------------------------------------------------------------------------------------------------
# Collection loop
# --------------------------------------------------------------------------------------------------------------------------
@dataclass
class RunState:
    rows: dict = field(default_factory=dict)         # review_id -> record
    clicks: int = 0
    log: list = field(default_factory=list)
    stop_reason: str = ""


def collect_list(driver, cfg, st, sort_label, max_clicks=None, verified_pass=False):
    """Parse the visible reviews, click 'Show More' (up to max_clicks, or until it disappears) and keep parsing."""
    seen, clicks = 0, 0
    while True:
        items = driver.find_elements(By.CSS_SELECTOR, SEL_ITEM)
        for pos, el in enumerate(items[seen:], start=seen + 1):
            rec = with_retries(lambda: parse_review(el.get_attribute("outerHTML"), cfg.sku), log=st.log)
            rid = rec["review_id"]
            if rid not in st.rows:
                if len(st.rows) >= cfg.max_reviews:
                    st.stop_reason = "max_reviews reached"
                    return
                st.rows[rid] = {**rec, "verified_buyer": False}
            if verified_pass:
                st.rows[rid]["verified_buyer"] = True
            else:
                st.rows[rid][f"rank_{sort_label.lower().replace(' ', '_')}"] = pos
        seen = len(items)
        if max_clicks is not None and clicks >= max_clicks:
            return
        if st.clicks >= cfg.max_total_clicks:
            st.stop_reason = "max_total_clicks budget reached"
            return
        if not with_retries(lambda: click_show_more(driver), log=st.log):
            return                                                       # no 'Show More' left -> list exhausted
        clicks += 1
        st.clicks += 1
        try:
            WebDriverWait(driver, 20).until(lambda d: len(d.find_elements(By.CSS_SELECTOR, SEL_ITEM)) > seen)
        except TimeoutException:
            return
        polite_sleep(cfg)
        check_not_blocked(driver)


def scrape(cfg):
    """Full run. Returns (DataFrame, run_log dict)."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    allowed, robots_txt = robots_allows(cfg.product_url.rstrip("/") + "/review")
    run = {"product_url": cfg.product_url, "sku": cfg.sku, "robots_allows_reviews_page": allowed, "started_utc": started.isoformat(),
           "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}}
    if not allowed:
        raise PermissionError("robots.txt disallows this URL - not scraping.")
    st = RunState()
    driver = make_driver(cfg)
    try:
        open_reviews_page(driver, cfg, st.log)
        # Pass 1 - the sort we page to exhaustion (catches every review)
        apply_sort(driver, cfg, cfg.full_sort, st.log)
        collect_list(driver, cfg, st, cfg.full_sort, max_clicks=None)
        n_full = len(st.rows)
        # Pass 2 - the other sort orders, first page(s) only: records how each review ranks under Most Helpful / Highest / Lowest / Most Relevant ...
        for label in [s for s in cfg.sorts if s != cfg.full_sort]:
            if st.stop_reason:
                break
            apply_sort(driver, cfg, label, st.log)
            collect_list(driver, cfg, st, label, max_clicks=cfg.shallow_clicks)
        # Pass 3 - 'Verified Buyers Only' filter: flags verified purchases
        if cfg.verified_pass and not st.stop_reason:
            apply_sort(driver, cfg, cfg.full_sort, st.log)
            set_verified_only(driver, cfg, True, st.log)
            collect_list(driver, cfg, st, cfg.full_sort, max_clicks=None, verified_pass=True)
    except BlockedError as e:
        st.stop_reason = f"BLOCKED: {e}"
    finally:
        driver.quit()
    df = pd.DataFrame(st.rows.values())
    run.update({"finished_utc": datetime.now(timezone.utc).isoformat(), "unique_reviews": len(df), "show_more_clicks": st.clicks,
                "stop_reason": st.stop_reason or "completed (all lists exhausted)", "events": st.log})
    if len(df):
        df.insert(1, "product_sku", cfg.sku)
        df.insert(2, "product_name", cfg.product_name)
        df["source"] = "Best Buy Canada (bestbuy.ca)"
        df["review_url"] = cfg.product_url.rstrip("/") + "/review"
        df["scraped_at_utc"] = run["finished_utc"]
    return df, run
