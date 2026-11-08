"""Stage 4: match identified coins to Numista types and fetch prices (free API plan).

    uv run numista.py --status            # quota used this month, match statistics (no requests)
    uv run numista.py                     # process coins in priority order until the budget is used
    uv run numista.py --max-requests 300  # use at most 300 new requests in this run

Every API response is cached in results/numista_cache/, so re-running only spends
requests on coins that are not resolved yet. The monthly counter lives in
results/numista_usage.json; the free plan allows 2000 requests per calendar month.

Requests per coin type: 1 search per (issuer, year) - shared by all coins of that
country and year - plus details, issues and prices per Numista type.
Output: <run>/numista_matches.jsonl (rewritten on every run, one row per coin).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import requests

from coins_common import DEFAULT_RUN, Item, apply_overrides, crop_side, read_jsonl

API = "https://api.numista.com/v3"
CACHE_DIR = "results/numista_cache"
USAGE_FILE = "results/numista_usage.json"
MONTHLY_QUOTA = 2000
MONTHLY_CAP = 1950  # hard stop per calendar month; 50 spare for safety
COIN_OBJECT_TYPE = 147  # "coins only" (includes commemoratives)

# LLM country names that fuzzy matching gets wrong -> Numista issuer code
ISSUER_ALIASES = {
    "west germany": "allemagne",
    "germany federal republic": "allemagne",
    "federal republic of germany": "allemagne",
    "east germany": "ddr",
    "german democratic republic": "ddr",
    "great britain": "royaume-uni",
    "uk": "royaume-uni",
    "usa": "etats-unis",
    "united states of america": "etats-unis",
    "czechoslovak socialist republic": "tchecoslovaquie",
    "russian sfsr": "rsfsr",
    "rsfsr": "rsfsr",
    "ussr": "ancienne_urss",
    "kingdom of serbs croats and slovenes": "yugoslavia_section",
    "austria hungary": "austria",  # Numista files these under the Austrian Empire
    "saarland": "sarre",
    "saar": "sarre",
    # German Empire states (the plain names would match duchies/notgeld)
    "prussia": "prussia_kingdom_section",
    "saxony": "saxony_kingdom",
    "bavaria": "bavaria_kingdom",
    "baden": "baden_grand_duchy",
    "wurttemberg": "wurttemberg_kingdom",
    "hamburg": "hamburg_hanseatic_city",
    "hanover": "hannover_kingdom",
    "hannover": "hannover_kingdom",
    "british india": "india_british_section",
    "burma": "myanmar_section",
    "united arab republic": "egypte",
    "saint thomas and prince": "sao_tome-et-principe",
    "british caribbean territories": "oeco",
}
# German states changed rank in 1806: (last year, issuer code), first match wins
YEAR_ALIASES = {
    "bavaria": [(1805, "bavaria_electorate"), (9999, "bavaria_kingdom")],
    "saxony": [(1805, "saxony_electorate"), (9999, "saxony_kingdom")],
}
FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125}
STOPWORDS = {"the", "of", "and", "a", "an", "in", "on", "for", "to", "de", "der", "die"}


class QuotaReached(Exception):
    pass


# --------------------------------------------------------------------------- API


class NumistaClient:
    """API client with a permanent disk cache and a monthly request budget."""

    def __init__(self, budget: int) -> None:
        key = os.environ.get("NUMISTA_API_KEY")
        if not key:
            sys.exit("NUMISTA_API_KEY missing in .env")
        self.session = requests.Session()
        self.session.headers["Numista-API-Key"] = key
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.month = date.today().strftime("%Y-%m")
        self.usage = load_usage()
        self.budget = budget
        self.spent = 0

    def used_this_month(self) -> int:
        return self.usage.get(self.month, 0)

    @staticmethod
    def _cache_path(path: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        params = {k: v for k, v in sorted(params.items()) if v is not None}
        params.setdefault("lang", "en")
        key = hashlib.sha1(
            json.dumps([path, params], sort_keys=True).encode()
        ).hexdigest()
        return os.path.join(CACHE_DIR, f"{key}.json"), params

    def cached(self, path: str, **params: Any) -> Any:
        """Cached response or None; never sends a request."""
        cache_path, _ = self._cache_path(path, params)
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)["response"]

    def get(self, path: str, **params: Any) -> Any:
        cache_path, params = self._cache_path(path, params)
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)["response"]

        if self.spent >= self.budget or self.used_this_month() >= MONTHLY_CAP:
            raise QuotaReached()
        response = self.session.get(f"{API}{path}", params=params, timeout=30)
        self.spent += 1
        self.usage[self.month] = self.used_this_month() + 1
        save_usage(self.usage)
        time.sleep(0.25)  # no parallel requests: the API rejects simultaneous ones
        if response.status_code == 429:
            raise QuotaReached()
        if response.status_code in (400, 404):  # cached too: never ask twice
            body: Any = None
        else:
            response.raise_for_status()
            body = response.json()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"path": path, "params": params, "response": body}, f)
        return body


def load_usage() -> Dict[str, int]:
    if not os.path.exists(USAGE_FILE):
        return {}
    with open(USAGE_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_usage(usage: Dict[str, int]) -> None:
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(usage, f, indent=2)


# --------------------------------------------------------------------- matching


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"\(.*?\)", " ", text.lower())
    return " ".join(re.findall(r"[a-z0-9]+", text))


def words(text: Optional[str]) -> set[str]:
    return {w for w in normalize(text or "").split() if w not in STOPWORDS}


class IssuerIndex:
    def __init__(self, client: NumistaClient) -> None:
        self.issuers = client.get("/issuers")["issuers"]
        # Notgeld (emergency money) issuers share names with the states; keep them apart
        self.by_name: Dict[str, List[Item]] = {}
        self.notgeld_by_name: Dict[str, List[Item]] = {}
        for issuer in self.issuers:
            is_notgeld = "notgeld" in issuer["code"] or "notgeld" in (
                issuer.get("parent") or {}
            ).get("code", "")
            index = self.notgeld_by_name if is_notgeld else self.by_name
            # "Germany, Federal Republic of" -> also "germany federal republic"
            # "Bahamas, The" -> also "bahamas"
            full = issuer["name"]
            for name in {full, full.replace(", ", " "), full.split(",")[0]}:
                index.setdefault(normalize(name), []).append(issuer)

    @staticmethod
    def variants(country: str) -> List[str]:
        """'Germany (Empire) - Prussia' -> ['germany empire prussia', 'germany prussia',
        'prussia', 'empire', 'germany']: most to least specific."""
        keep_parens = normalize(country.replace("(", " ").replace(")", " "))
        segments = [s for s in re.split(r"\s+[-–]\s+", country) if s.strip()]
        in_parens = re.findall(r"\((.*?)\)", country)
        out = [keep_parens, normalize(country)]
        out += [normalize(s) for s in reversed(segments)]
        out += [normalize(p) for p in in_parens]
        return list(dict.fromkeys(v for v in out if v))

    def lookup(
        self, country: Optional[str], year: Optional[int] = None
    ) -> Optional[str]:
        if not country:
            return None
        index = self.notgeld_by_name if "notgeld" in country.lower() else self.by_name
        candidates = self.variants(country)
        matches = None
        for name in candidates:
            if name in YEAR_ALIASES and year:
                return next(code for last, code in YEAR_ALIASES[name] if year <= last)
            if name in ISSUER_ALIASES:
                return ISSUER_ALIASES[name]
            if name in index:
                matches = index[name]
                break
        if not matches:
            # Strict cutoff: 0.8 already matched "saarland" to "samarqand"
            close = difflib.get_close_matches(
                candidates[0], index.keys(), n=1, cutoff=0.88
            )
            if not close:
                return None
            matches = index[close[0]]
        # Broadest node: a group code also returns the types of its child issuers
        return min(matches, key=lambda i: i.get("level", 9))["code"]


def title_value(title: str) -> Optional[float]:
    """'½ Franc (Helvetia)' -> 0.5, '100 Korun (...)' -> 100, '1½ Pence' -> 1.5."""
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)?\s*([½¼¾⅓⅔⅛])?(?:\s*/\s*(\d+))?", title)
    if not m or not (m.group(1) or m.group(2)):
        return None
    value = float(m.group(1).replace(",", ".")) if m.group(1) else 0.0
    if m.group(3):  # "1/2 Dollar"
        return value / float(m.group(3))
    return value + FRACTIONS.get(m.group(2) or "", 0.0)


def split_title(title: str) -> Tuple[str, str, str]:
    """'1 Dollar "Morgan Dollar"' / '5 Mark - Louis II' / '50 Korun (Kremnica Mint)'
    -> (denomination part, subject part from quotes/parentheses, ruler part)."""
    denomination = re.split(r"\s[-–]\s|\(|\"|“", title)[0]
    subject = " ".join(
        re.findall(r"\((.*?)\)", title) + re.findall(r"[\"“](.*?)[\"”]", title)
    )
    ruler_match = re.search(r"\s[-–]\s([^(\"“]*)", title)
    return denomination, subject, ruler_match.group(1) if ruler_match else ""


def same_unit(a: set[str], b: set[str]) -> bool:
    """'rouble' ~ 'ruble', 'korun' ~ 'koruna', 'zlotych' ~ 'zlotys'; 'lira' !~ 'kurus'."""
    return any(
        x[:4] == y[:4] or difflib.SequenceMatcher(None, x, y).ratio() >= 0.75
        for x in a
        for y in b
    )


def text_score(ident: Item, candidate: Item) -> float:
    title = candidate["title"]
    denomination, subject, ruler = split_title(title)
    score = 0.0
    value = title_value(title)
    if value is not None and ident.get("denomination_value") is not None:
        score += 3.0 if abs(value - float(ident["denomination_value"])) < 1e-6 else -3.0
    unit_words = {w for w in words(ident.get("denomination_title")) if not w.isdigit()}
    title_units = {w for w in words(denomination) if not w.isdigit()}
    if unit_words and title_units:
        # same number but another unit (10 Pfennig vs 10 Mark) is a different coin
        score += 1.0 if same_unit(unit_words, title_units) else -2.0
    wanted = words(ident.get("subject")) | words(ident.get("numista_query"))
    wanted -= words(ident.get("denomination_title"))
    subject_words = words(subject) - title_units
    if wanted and subject_words:
        score += 2.0 * len(wanted & subject_words) / len(subject_words)
    elif not ident.get("subject") and not subject_words:
        score += 0.5  # plain circulation coin vs plain title
    if wanted & words(ruler):
        score += 0.5
    if re.search(r"pattern|trial|essai|probe|restrike|fantasy", title, re.I):
        score -= 1.5
    return score


class VisualRanker:
    """Tie-breaker: compare our crops with Numista thumbnails (not API requests)."""

    def __init__(self) -> None:
        self.model: Any = None

    def _load(self) -> None:
        from clustering import get_feature_extractor

        self.model, self.preprocess, self.device = get_feature_extractor()

    def thumbnail(self, url: str):
        import cv2
        import numpy as np

        path = os.path.join(
            CACHE_DIR, "thumbs", hashlib.sha1(url.encode()).hexdigest() + ".jpg"
        )
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            response = requests.get(
                url, timeout=30, headers={"User-Agent": "coin-catalogue"}
            )
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

    def similarity(self, coin: Item, candidate: Item) -> float:
        from clustering import embed_images

        if self.model is None:
            self._load()
        urls = [candidate.get("obverse_thumbnail"), candidate.get("reverse_thumbnail")]
        if not all(urls):
            return 0.0
        images = [crop_side(coin, "front"), crop_side(coin, "back")]
        try:
            thumbs = [self.thumbnail(u) for u in urls]
        except requests.RequestException:
            return 0.0  # picture unavailable: decide on text alone
        if any(t is None for t in thumbs):
            return 0.0
        images += thumbs
        a, b, obv, rev = embed_images(images, self.model, self.preprocess, self.device)
        return float(max((a @ obv + b @ rev) / 2, (a @ rev + b @ obv) / 2))


def search_candidates(
    client: NumistaClient, issuers: IssuerIndex, ident: Item
) -> Tuple[Optional[str], List[Item]]:
    year = re.match(r"\d{3,4}", ident.get("year") or "")
    issuer = issuers.lookup(ident.get("country"), int(year.group(0)) if year else None)
    params: Dict[str, Any] = {"object_type": COIN_OBJECT_TYPE, "count": 100}
    if issuer and year:
        result = client.get("/types", issuer=issuer, date=year.group(0), **params)
        if result and result["count"] > 100 and ident.get("denomination_title"):
            result = client.get(
                "/types",
                issuer=issuer,
                date=year.group(0),
                q=ident["denomination_title"],
                **params,
            )
    elif issuer:
        result = client.get(
            "/types", issuer=issuer, q=ident.get("numista_query"), **params
        )
    elif ident.get("numista_query"):
        result = client.get(
            "/types",
            q=ident.get("numista_query"),
            date=year.group(0) if year else None,
            **params,
        )
    else:
        return issuer, []  # nothing to search for (the API needs q, issuer or date)
    return issuer, (result or {}).get("types", [])


def pick_issue(issues: List[Item], ident: Item) -> Optional[Item]:
    if not issues:
        return None
    year = re.match(r"\d{3,4}", ident.get("year") or "")
    same_year = [
        i for i in issues if year and str(i.get("gregorian_year")) == year.group(0)
    ]
    pool = same_year or (issues if len(issues) == 1 else [])
    mint = (ident.get("mint_mark") or "").strip().upper()
    if mint:
        with_mint = [i for i in pool if (i.get("mint_letter") or "").upper() == mint]
        pool = with_mint or pool
    else:
        without_mint = [i for i in pool if not i.get("mint_letter")]
        pool = without_mint or pool
    return pool[0] if pool else None


def resolve(
    client: NumistaClient,
    issuers: IssuerIndex,
    ranker: VisualRanker,
    ident: Item,
    coin: Item,
) -> Item:
    row: Item = {"coin_id": ident["coin_id"]}
    type_id = ident.get("numista_type_id")
    best: Item = {}
    if type_id is None:
        issuer, candidates = search_candidates(client, issuers, ident)
        row["issuer_code"] = issuer
        if not candidates:
            return {**row, "status": "no_candidates"}
        scored = sorted(
            ((text_score(ident, c), c) for c in candidates), key=lambda sc: -sc[0]
        )
        top = [sc for sc in scored if sc[0] >= scored[0][0] - 0.5][:5]
        if len(top) > 1:  # ambiguous on text alone -> let the pictures decide
            top = sorted(
                ((s + 2.0 * ranker.similarity(coin, c), c) for s, c in top),
                key=lambda sc: -sc[0],
            )
            scored = top + [sc for sc in scored if sc not in top]
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else best_score - 3
        row.update(
            match_score=round(best_score, 2),
            match_margin=round(best_score - runner_up, 2),
            alternatives=[
                {"id": c["id"], "title": c["title"], "score": round(s, 2)}
                for s, c in scored[1:4]
            ],
        )
        type_id = best["id"]

    # Type details (weight, composition, KM#) cost a request; base-metal coins don't
    # need them for valuation, so they use the search result unless already cached.
    precious = priority(ident)[0] == 0
    if not best or precious or (ident.get("est_value_eur_guess") or 0) >= 5:
        details = client.get(f"/types/{type_id}")
    else:
        details = client.cached(f"/types/{type_id}") or {
            "title": best.get("title"),
            "issuer": best.get("issuer"),
            "min_year": best.get("min_year"),
            "max_year": best.get("max_year"),
        }
    if not details:
        return {**row, "status": "type_not_found", "type_id": type_id}
    references = details.get("references") or []
    row.update(
        status="matched",
        type_id=type_id,
        title=details.get("title"),
        url=details.get("url")
        or f"https://en.numista.com/catalogue/pieces{type_id}.html",
        issuer_name=(details.get("issuer") or {}).get("name"),
        value_text=(details.get("value") or {}).get("text"),
        numeric_value=(details.get("value") or {}).get("numeric_value"),
        currency=((details.get("value") or {}).get("currency") or {}).get("name"),
        composition=(details.get("composition") or {}).get("text"),
        weight_g=details.get("weight"),
        size_mm=details.get("size"),
        km=", ".join(
            f"{r['catalogue']['code']}# {r['number']}"
            for r in references
            if r.get("catalogue", {}).get("code") == "KM"
        )
        or None,
        min_year=details.get("min_year"),
        max_year=details.get("max_year"),
    )

    issue_id = ident.get("numista_issue_id")
    issues = client.get(f"/types/{type_id}/issues") or []
    issue = next((i for i in issues if i["id"] == issue_id), None) or pick_issue(
        issues, ident
    )
    if issue is None:
        return {**row, "status": "matched_no_issue"}
    row.update(
        issue_id=issue["id"],
        issue_year=issue.get("gregorian_year") or issue.get("year"),
        mint_letter=issue.get("mint_letter"),
        mintage=issue.get("mintage"),
    )
    prices = client.get(f"/types/{type_id}/issues/{issue['id']}/prices", currency="EUR")
    row["prices"] = {p["grade"]: p["price"] for p in (prices or {}).get("prices", [])}
    return row


# -------------------------------------------------------------------------- main


def priority(ident: Item) -> Tuple[int, float]:
    precious = bool(
        re.search(
            r"silver|gold|platinum|palladium",
            ident.get("composition_guess") or "",
            re.I,
        )
    )
    return (0 if precious else 1, -(ident.get("est_value_eur_guess") or 0.0))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--max-requests", type=int, default=MONTHLY_CAP)
    parser.add_argument("--status", action="store_true", help="only print statistics")
    parser.add_argument(
        "--min-guess",
        type=float,
        default=0.0,
        help="skip non-precious coins whose LLM value guess is below this (EUR)",
    )
    args = parser.parse_args()  # fmt: skip

    out_path = os.path.join(args.run, "numista_matches.jsonl")
    usage = load_usage()
    month = date.today().strftime("%Y-%m")
    print(
        f"Numista requests used in {month}: {usage.get(month, 0)} / {MONTHLY_QUOTA} (script stops at {MONTHLY_CAP})"
    )
    if args.status:
        rows = read_jsonl(out_path)
        print(Counter(r["status"] for r in rows).most_common())
        return 0

    from coins_common import load_coins

    coins = {c["coin_id"]: c for c in load_coins(args.run)}
    idents = {
        r["coin_id"]: r
        for r in read_jsonl(os.path.join(args.run, "identifications.jsonl"))
    }
    if not idents:
        sys.exit("No identifications yet - run identify.py first.")
    apply_overrides(idents)

    budget = min(args.max_requests, MONTHLY_CAP - usage.get(month, 0))
    client = NumistaClient(budget=max(budget, 0))
    issuers = IssuerIndex(client)
    ranker = VisualRanker()

    rows: List[Item] = []
    for ident in sorted(idents.values(), key=priority):
        cid = ident["coin_id"]
        if ident.get("exclude") or not ident.get("is_coin", True):
            rows.append({"coin_id": cid, "status": "excluded"})
            continue
        precious = priority(ident)[0] == 0
        if not precious and (ident.get("est_value_eur_guess") or 0) < args.min_guess:
            rows.append({"coin_id": cid, "status": "skipped_low_value"})
            continue
        # Keep going after the budget is used up: coins whose lookups are all cached
        # still resolve; only coins that need a new request wait (QuotaReached).
        try:
            rows.append(resolve(client, issuers, ranker, ident, coins[cid]))
        except QuotaReached:
            rows.append({"coin_id": cid, "status": "pending_quota"})
        except requests.RequestException as e:  # one bad coin must not stop the run
            print(f"{cid}: {e}", file=sys.stderr)
            rows.append({"coin_id": cid, "status": "api_error", "error": str(e)})

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(r["status"] for r in rows)
    print(
        f"New requests this run: {client.spent}; used in {month}: {client.used_this_month()}"
    )
    print("Status:", dict(counts.most_common()))
    if counts.get("pending_quota"):
        print(
            f"{counts['pending_quota']} coins wait for requests (run budget or monthly cap "
            f"reached). Cached lookups are free; the rest continues next month."
        )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
