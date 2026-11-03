"""
Scrape Google Maps rating + review count for POIs in Da Nang, then rank them
with a Bayesian weighted score.

Requires:  pip install playwright && playwright install chromium
Input:     pois.py exposing POIS = [{"poi_id", "name", "category", "coordinates": [lat, lng]}, ...]
Output:    ranked_pois.csv (plus metrics_cache.json so reruns skip finished POIs)
"""

import csv
import json
import os
import random
import re
from pathlib import Path
from urllib.parse import quote_plus

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

from pois import POIS

# ---------------------------------------------------------------- settings
HEADLESS = False          # keep False for the first run so you can see what happens
LIMIT = None              # e.g. 3 to test on the first 3 POIs, None for all
BASE_DIR = Path(__file__).resolve().parent  # files are read/written next to this script
CACHE_FILE = BASE_DIR / "metrics_cache.json"
OUTPUT_CSV = BASE_DIR / "ranked_pois.csv"
CITY_SUFFIX = "Đà Nẵng"
# Hand-entered values, used instead of scraping (keyed by poi_id)
MANUAL_OVERRIDES = {
    1: {"rating": 4.5, "review_count": 9905},    # Cầu Rồng
    46: {"rating": 4.1, "review_count": 10140},  # GO! Đà Nẵng
}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------- parsing
def parse_int(s):
    digits = re.sub(r"\D", "", s or "")
    return int(digits) if digits else 0


def parse_rating(s):
    m = re.search(r"([1-5][\.,]\d)", s or "")
    return float(m.group(1).replace(",", ".")) if m else 0.0


# ---------------------------------------------------------------- scraping
def open_place_page(page):
    """Maps either redirects straight to the place or shows a results list."""
    place_url = re.compile(r"/maps/place/")
    try:
        page.wait_for_url(place_url, timeout=8000)
        return
    except PWTimeout:
        pass

    # Results list: click the first hit
    page.locator("div[role='feed'] a[href*='/maps/place/']").first.click(timeout=8000)
    page.wait_for_url(place_url, timeout=8000)


def read_metrics(page):
    page.wait_for_selector(
        "div.F7nice, span[role='img'][aria-label*='sao']", timeout=10000
    )

    rating, reviews = 0.0, 0

    # Preferred: rating block text, e.g. "4,5\n(9.905)"
    block = page.query_selector("div.F7nice")
    if block:
        text = block.inner_text()
        rating = parse_rating(text)
        m = re.search(r"\(([\d\.,]+)\)", text)
        if m:
            reviews = parse_int(m.group(1))

    # Fallback: aria-labels like "4,5 sao" and "9.905 bài đánh giá"
    if rating == 0.0:
        el = page.query_selector("span[role='img'][aria-label*='sao']")
        rating = parse_rating(el.get_attribute("aria-label")) if el else 0.0
    if reviews == 0:
        el = page.query_selector(
            "[aria-label*='bài đánh giá'], [aria-label*='đánh giá']"
        )
        reviews = parse_int(el.get_attribute("aria-label")) if el else 0

    h1 = page.query_selector("h1")
    matched_name = h1.inner_text().strip() if h1 else ""
    return rating, reviews, matched_name


def scrape_one(page, poi):
    lat, lng = poi.get("coordinates", [16.06, 108.22])
    url = (
        "https://www.google.com/maps/search/"
        f"{quote_plus(poi['name'] + ' ' + CITY_SUFFIX)}/@{lat},{lng},15z?hl=vi"
    )
    page.goto(url, timeout=20000, wait_until="domcontentloaded")

    # Consent dialog (only appears for some regions/IPs)
    try:
        page.click(
            "button:has-text('Chấp nhận tất cả'), button:has-text('Accept all')",
            timeout=2000,
        )
    except PWTimeout:
        pass

    open_place_page(page)
    return read_metrics(page)


def fetch_gmaps_metrics(pois_list):
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)

    targets = pois_list[:LIMIT] if LIMIT else pois_list
    print(f"Scraping {len(targets)} POIs ({len(cache)} already cached)...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(locale="vi-VN", user_agent=USER_AGENT)
        page = context.new_page()

        for idx, poi in enumerate(targets, 1):
            key = str(poi.get("poi_id") or poi["name"])
            tag = f"[{idx}/{len(targets)}] {poi['name']}"

            if poi.get("poi_id") in MANUAL_OVERRIDES:
                poi.update(MANUAL_OVERRIDES[poi["poi_id"]])
                poi["matched_name"] = "(manual)"
                print(f"{tag} -> manual ({poi['rating']} ⭐, {poi['review_count']})")
                continue

            if key in cache and cache[key]["rating"] > 0:
                poi.update(cache[key])
                print(f"{tag} -> cached ({poi['rating']} ⭐, {poi['review_count']})")
                continue

            rating, reviews, matched = 0.0, 0, ""
            for attempt in (1, 2):  # one retry
                try:
                    rating, reviews, matched = scrape_one(page, poi)
                    if rating > 0 and reviews > 0:
                        break
                except Exception as err:
                    print(f"{tag} attempt {attempt} failed: {str(err).splitlines()[0]}")
                    print(f"   URL: {page.url}")
                    page.screenshot(path=str(BASE_DIR / f"debug_{idx}.png"))
                page.wait_for_timeout(2000)

            # Never keep a half-parsed result
            if rating == 0.0 or reviews == 0:
                rating, reviews = 0.0, 0

            poi["rating"] = rating
            poi["review_count"] = reviews
            poi["matched_name"] = matched
            print(f"{tag} -> {rating} ⭐ ({reviews} reviews) | matched: {matched!r}")

            if rating > 0:
                cache[key] = {
                    "rating": rating,
                    "review_count": reviews,
                    "matched_name": matched,
                }
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)

            page.wait_for_timeout(random.randint(1500, 3500))

        browser.close()
    return targets


# ---------------------------------------------------------------- scoring
def calculate_bayesian_scores(pois_list):
    """Bayesian weighted score: (v/(v+m))*R + (m/(v+m))*C."""
    valid = [p for p in pois_list if p.get("review_count", 0) > 0]
    if not valid:
        print("\nWarning: no review data collected.")
        return pois_list

    C = sum(p["rating"] for p in valid) / len(valid)
    m = sum(p["review_count"] for p in valid) / len(valid)
    print(f"\nBayesian baseline -> mean rating C={C:.2f}, mean reviews m={m:.0f}\n")

    for poi in pois_list:
        v, R = poi.get("review_count", 0), poi.get("rating", 0.0)
        poi["score"] = round((v / (v + m)) * R + (m / (v + m)) * C, 3) if v > 0 else 0.0

    return sorted(pois_list, key=lambda x: x["score"], reverse=True)


# ---------------------------------------------------------------- export
def export_to_csv(pois_list, filename=OUTPUT_CSV):
    fieldnames = [
        "poi_id", "name", "matched_name", "category",
        "latitude", "longitude", "rating", "review_count", "score",
    ]
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for poi in pois_list:
            lat, lng = poi.get("coordinates", [None, None])
            writer.writerow({
                "poi_id": poi.get("poi_id"),
                "name": poi.get("name"),
                "matched_name": poi.get("matched_name", ""),
                "category": poi.get("category"),
                "latitude": lat,
                "longitude": lng,
                "rating": poi.get("rating", 0.0),
                "review_count": poi.get("review_count", 0),
                "score": poi.get("score", 0.0),
            })
    print(f"Exported {len(pois_list)} POIs to '{filename}'")


if __name__ == "__main__":
    scraped = fetch_gmaps_metrics(POIS)
    ranked = calculate_bayesian_scores(scraped)
    export_to_csv(ranked)