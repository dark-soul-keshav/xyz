"""Ledger Agent: answers plain-English questions about the Acme order ledger.

Pure standard library. Configuration (environment variables):
  LEDGER_ROOT_URL  required  the root API URL from the exam page (the one ending path=/)
  LLM_API_KEY      optional  enables LLM question parsing (OpenAI-compatible API)
  LLM_BASE_URL     optional  default https://api.openai.com/v1
  LLM_MODEL        optional  default gpt-4o-mini
  PORT             optional  default 8000
Without LLM_API_KEY a rule-based parser is used.
"""
import difflib
import json
import os
import re
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

IST = timezone(timedelta(hours=5, minutes=30))
ROOT_URL = os.environ.get("LEDGER_ROOT_URL", "")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

STATE = {"records": [], "meta": {}, "ready": threading.Event(), "error": None}

MONTHS_FULL = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
               "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}
MONTHS_ABBR = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
               "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12}

# ----------------------------------------------------------------- loading


def http_json(url, tries=4, timeout=30):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ledger-agent"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # retry
            last = e
            time.sleep(0.5 * (i + 1))
    raise last


def parse_dt(s):
    s = str(s).strip().replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        s2 = re.sub(r"(\.\d+)", lambda m: (m.group(1) + "000000")[:7], s)
        d = datetime.fromisoformat(s2)
    if d.tzinfo is None:
        d = d.replace(tzinfo=IST)
    return d


def build_records(raw_rows, rates):
    """Dedupe by id (latest updated_at wins), convert to USD and IST dates."""
    best = {}
    for row in raw_rows:
        oid = str(row.get("id", "")).strip().upper()
        if not oid:
            continue
        try:
            upd = parse_dt(row.get("updated_at") or row.get("created_at"))
        except Exception:
            upd = datetime.min.replace(tzinfo=timezone.utc)
        cur = best.get(oid)
        if cur is None or upd >= cur[0]:
            best[oid] = (upd, row)
    records = []
    for oid, (_, row) in best.items():
        cur = str(row.get("currency", "USD")).strip().upper()
        rate = rates.get(cur)
        if rate is None:
            rate = 1.0
        try:
            amount = float(row.get("amount"))
        except (TypeError, ValueError):
            amount = float(row.get("qty", 0)) * float(row.get("unit_price", 0))
        created = parse_dt(row["created_at"]).astimezone(IST)
        records.append({
            "id": oid,
            "customer": str(row.get("customer", "")).strip(),
            "region": str(row.get("region", "")).strip(),
            "product": str(row.get("product", "")).strip(),
            "qty": float(row.get("qty") or 0),
            "currency": cur,
            "status": str(row.get("status", "")).strip().lower(),
            "usd": amount * rate,
            "date": created.date(),
        })
    return records


def compute_meta(records, n_raw):
    dates = [r["date"] for r in records]
    return {
        "orders_after_dedupe": len(records),
        "rows_fetched": n_raw,
        "statuses": sorted({r["status"] for r in records}),
        "regions": sorted({r["region"] for r in records}),
        "products": sorted({r["product"] for r in records}),
        "currencies": sorted({r["currency"] for r in records}),
        "first_date": str(min(dates)) if dates else None,
        "last_date": str(max(dates)) if dates else None,
        "dominant_year": Counter(d.year for d in dates).most_common(1)[0][0] if dates else 2026,
    }


def load_data():
    root = http_json(ROOT_URL)
    links = root["links"]
    rates = http_json(links["rates"])["usd_per_unit"]
    orders_url = links["orders"]
    first = http_json(orders_url)
    rows = list(first.get("orders", []))
    pages = first.get("pages")
    if pages:
        def page_url(n):
            return re.sub(r"([?&]page=)\d+", lambda m: m.group(1) + str(n), orders_url)

        with ThreadPoolExecutor(12) as ex:
            for data in ex.map(lambda n: http_json(page_url(n)), range(2, int(pages) + 1)):
                rows.extend(data.get("orders", []))
    else:
        nxt = first.get("next")
        while nxt:
            data = http_json(nxt)
            rows.extend(data.get("orders", []))
            nxt = data.get("next")
    records = build_records(rows, rates)
    STATE["records"] = records
    STATE["meta"] = compute_meta(records, len(rows))


def loader_thread():
    while True:
        try:
            load_data()
            STATE["error"] = None
            STATE["ready"].set()
            time.sleep(1800)  # refresh every 30 minutes
        except Exception as e:
            STATE["error"] = repr(e)
            time.sleep(5)


# ----------------------------------------------------------------- query engine

METRICS = {"revenue_usd", "order_count", "units_sold", "avg_order_value",
           "refund_amount_usd", "distinct_customers"}
DEFAULT_STATUS = {"revenue_usd": "paid", "units_sold": "paid", "avg_order_value": "paid",
                  "refund_amount_usd": "refunded", "order_count": None, "distinct_customers": None}


def canon(value, options):
    if value is None:
        return None
    v = str(value).strip().lower()
    for o in options:
        if o.lower() == v:
            return o
    m = difflib.get_close_matches(v, [o.lower() for o in options], n=1, cutoff=0.8)
    if m:
        for o in options:
            if o.lower() == m[0]:
                return o
    return str(value)


def to_date(s):
    if not s:
        return None
    if isinstance(s, date):
        return s
    return date.fromisoformat(str(s)[:10])


def metric_value(rows, metric):
    if metric in ("revenue_usd", "refund_amount_usd"):
        return sum(r["usd"] for r in rows)
    if metric == "order_count":
        return len(rows)
    if metric == "units_sold":
        total = sum(r["qty"] for r in rows)
        return int(total) if total == int(total) else total
    if metric == "avg_order_value":
        return sum(r["usd"] for r in rows) / len(rows) if rows else 0.0
    if metric == "distinct_customers":
        return len({r["customer"] for r in rows})
    raise ValueError("unknown metric " + str(metric))


def fmt(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        return round(x, 2) + 0.0
    return x


def run_spec(spec):
    meta = STATE["meta"]
    metric = spec.get("metric") or "revenue_usd"
    if metric not in METRICS:
        raise ValueError("bad metric")
    status = spec.get("status")
    if status in (None, ""):
        status = DEFAULT_STATUS[metric]
    region = canon(spec.get("region"), meta["regions"])
    product = canon(spec.get("product"), meta["products"])
    currency = spec.get("currency")
    customer = spec.get("customer")
    start, end = to_date(spec.get("start")), to_date(spec.get("end"))

    rows = []
    for r in STATE["records"]:
        if status and str(status).lower() != "any" and r["status"] != str(status).lower():
            continue
        if region and r["region"] != region:
            continue
        if product and r["product"] != product:
            continue
        if currency and r["currency"] != str(currency).upper():
            continue
        if customer and r["customer"].lower() != str(customer).lower():
            continue
        if start and r["date"] < start:
            continue
        if end and r["date"] >= end:
            continue
        rows.append(r)

    group_by = spec.get("group_by")
    if group_by:
        groups = defaultdict(list)
        for r in rows:
            key = r["date"].strftime("%Y-%m") if group_by == "month" else r[group_by]
            groups[key].append(r)
        values = {k: metric_value(v, metric) for k, v in groups.items()}
        rank = spec.get("rank")
        if rank in ("top", "bottom") and values:
            ordered = sorted(values.items(), key=lambda kv: ((-kv[1] if rank == "top" else kv[1]), kv[0]))
            n = int(spec.get("n") or 1)
            if spec.get("return") == "value":
                return fmt(ordered[0][1])
            if n > 1:
                return [k for k, _ in ordered[:n]]
            return ordered[0][0]
        return {k: fmt(v) for k, v in sorted(values.items())}
    return fmt(metric_value(rows, metric))


# ----------------------------------------------------------------- rule-based parser


def month_range(y, m):
    return date(y, m, 1), date(y + (m == 12), m % 12 + 1, 1)


def parse_dates(q, default_year):
    ql = q.lower()
    iso = re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", ql)
    if len(iso) >= 2:
        a, b = date(*map(int, iso[0])), date(*map(int, iso[1]))
        return min(a, b), max(a, b) + timedelta(days=1)
    if len(iso) == 1:
        a = date(*map(int, iso[0]))
        return a, a + timedelta(days=1)
    m = re.search(r"\bq([1-4])\s*(?:of\s*)?(20\d\d)?", ql)
    if m:
        y = int(m.group(2) or default_year)
        q0 = int(m.group(1))
        s = date(y, 3 * q0 - 2, 1)
        e = date(y + (q0 == 4), (3 * q0) % 12 + 1, 1)
        return s, e
    names = "|".join(sorted(MONTHS_FULL, key=len, reverse=True))
    m = re.search(rf"\b({names})\b\.?,?\s*(?:of\s*)?(20\d\d)?", ql)
    if m:
        return month_range(int(m.group(2) or default_year), MONTHS_FULL[m.group(1)])
    abbr = "|".join(sorted(MONTHS_ABBR, key=len, reverse=True))
    m = re.search(rf"\b({abbr})\b\.?,?\s*(?:of\s*)?(20\d\d)\b", ql)
    if m:
        return month_range(int(m.group(2)), MONTHS_ABBR[m.group(1)])
    m = re.search(r"\b(20\d\d)\b", ql)
    if m:
        y = int(m.group(1))
        return date(y, 1, 1), date(y + 1, 1, 1)
    return None, None


def rule_parse(question):
    meta = STATE["meta"]
    q = question.lower()
    spec = {}

    for r in sorted(meta["regions"], key=len, reverse=True):
        if re.search(rf"\b{re.escape(r.lower())}\b", q):
            spec["region"] = r
            break
    for p in sorted(meta["products"], key=len, reverse=True):
        if re.search(rf"\b{re.escape(p.lower())}(?:s|es)?\b", q):
            spec["product"] = p
            break
    m = re.search(r"\bc\d{3,}\b", q)
    if m:
        spec["customer"] = m.group(0).upper()
    for s in meta["statuses"]:
        if s and (s in q or (len(s) >= 5 and s[:5] in q)):
            spec["status"] = s
            break
    cur = re.search(r"\b(usd|eur|inr)\b", q)
    if cur and re.search(r"\b(in|paid in|priced in)\s+(usd|eur|inr)\b.*\b(orders|invoices)\b", q):
        spec["currency"] = cur.group(1).upper()

    s, e = parse_dates(question, meta["dominant_year"])
    if s:
        spec["start"], spec["end"] = str(s), str(e)

    group = None
    dims = "product|region|customer|month"
    rank_cue = r"(top|best|highest|most|lowest|least|fewest|worst|biggest|largest|smallest|leading|bestselling|best-selling|top-selling|top-grossing|highest-grossing)"
    m_g = (re.search(rf"\b(which|what)\s+(?:[\w\-]+\s+){{0,3}}?({dims})s?\b", q)
           or re.search(rf"\b{rank_cue}\b[\w\-, ]{{0,30}}?\b({dims})s?\b", q))
    if m_g:
        group = m_g.group(m_g.lastindex)
        # a bare "what ... <dim>" without any ranking word is a filter, not a ranking
        if not re.search(rf"\b{rank_cue}\b", q) and not re.search(rf"\b(which)\s+(?:[\w\-]+\s+){{0,3}}?{group}s?\b", q):
            group = None
    metric = None
    if re.search(r"\b(average|mean|avg)\b", q):
        metric = "avg_order_value"
    elif re.search(r"\b(distinct|unique|different)\b.*\bcustomers?\b", q):
        metric = "distinct_customers"
    elif re.search(r"\b(units?|quantity|pieces|items)\b", q):
        metric = "units_sold"
    elif re.search(r"\b(how many|count|number of)\b", q):
        metric = "order_count"
    elif "refund" in q and re.search(r"\b(amount|value|total|worth|much)\b", q):
        metric = "refund_amount_usd"
        spec.setdefault("status", "refunded")
    else:
        metric = "revenue_usd"
    if group:
        spec["group_by"] = group
        spec["rank"] = "bottom" if re.search(r"\b(lowest|least|fewest|worst|smallest|minimum)\b", q) else "top"
        if re.search(r"\b(most|many)\s+orders\b", q):
            metric = "order_count"
        spec.pop(group, None)
    spec["metric"] = metric
    return spec


# ----------------------------------------------------------------- LLM parser

LLM_SYSTEM = """You translate a question about an order ledger into a JSON query. Reply with JSON only.
Keys (use null when not mentioned):
  metric: one of revenue_usd, order_count, units_sold, avg_order_value, refund_amount_usd, distinct_customers
  status: an order status value or "any" (leave null to use the default: paid for revenue/units/average, refunded for refund amount, any for counts)
  region, product, customer (like C1085), currency (USD/EUR/INR, only if the question restricts to orders placed in that currency)
  start, end: YYYY-MM-DD, start inclusive, end EXCLUSIVE, based on the order creation date (March 2026 -> 2026-03-01 to 2026-04-01)
  group_by: product, region, customer or month, only for "which X had the most/least ..." questions
  rank: top or bottom (with group_by); return: key (the name) or value
Known values: regions={regions}; products={products}; statuses={statuses}; currencies={currencies}. Data covers {first} to {last}.
"Revenue" means the sum of paid orders in USD. Refund questions use status refunded. Return only the JSON object."""


def llm_parse(question):
    meta = STATE["meta"]
    system = LLM_SYSTEM.format(regions=meta["regions"], products=meta["products"],
                               statuses=meta["statuses"], currencies=meta["currencies"],
                               first=meta["first_date"], last=meta["last_date"])
    body = json.dumps({
        "model": LLM_MODEL, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
    }).encode()
    req = urllib.request.Request(LLM_BASE + "/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + LLM_KEY})
    with urllib.request.urlopen(req, timeout=9) as r:
        out = json.loads(r.read().decode())
    return json.loads(out["choices"][0]["message"]["content"])


def answer_question(question):
    spec = None
    if LLM_KEY:
        try:
            spec = llm_parse(question)
            if spec.get("metric") not in METRICS:
                spec = None
        except Exception:
            spec = None
    if spec is None:
        spec = rule_parse(question)
    return run_spec(spec), spec


# ----------------------------------------------------------------- HTTP


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body=None):
        data = json.dumps(body).encode() if body is not None else b""
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        if body is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(204)

    def do_GET(self):
        self._send(200, {"ready": STATE["ready"].is_set(), "error": STATE["error"], **STATE["meta"]})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            question = str(body.get("question", ""))
            if not STATE["ready"].wait(12):
                return self._send(503, {"answer": None, "error": "data not loaded: " + str(STATE["error"])})
            ans, spec = answer_question(question)
            self._send(200, {"answer": ans})
        except Exception as e:
            self._send(500, {"answer": None, "error": repr(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if not ROOT_URL:
        raise SystemExit("Set LEDGER_ROOT_URL to the root API URL (the one ending path=/)")
    threading.Thread(target=loader_thread, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8000"))), Handler).serve_forever()
