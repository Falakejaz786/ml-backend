from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd

from db import get_db
from services.medicine_lookup import _fetch_all_medicines

@dataclass
class ShopCandidate:
    shop_id: str
    shop_name: str
    distance: float
    medicine_prices: Dict[str, float]
    medicine_quantities: Dict[str, int]


@dataclass
class ShopResult:
    shop_id: str
    shop_name: str
    distance: float
    covered_medicines: List[str]
    uncovered_medicines: List[str]
    coverage_ratio: float
    fulfillable_ratio: float
    total_price: float
    raw_score: float
    norm_score: float = 0.0


@dataclass
class MultiShopSolution:
    shops: List[ShopResult]
    total_price: float
    max_distance: float
    combined_score: float
    fully_covered: bool


def _load_shop_dataframe() -> pd.DataFrame:
   
    try:
        db = get_db()
        shops = list(db.shops.find({}))

        if not shops:
            return pd.DataFrame(columns=[
                "shop_id", "shop_name", "distance",
                "medicine_id", "price", "quantity",
            ])

        rows: List[Dict] = []
        for shop in shops:
            shop_id  = shop.get("shopId", str(shop["_id"]))
            name     = shop.get("name", "")
            distance = float(shop.get("distance_from_user", 0.0))

            for med_entry in shop.get("medicines", []):
                med_id = (
                    med_entry.get("medicine_id")
                    or med_entry.get("medicineId")
                    or med_entry.get("id")
                )
                if not med_id:
                    continue
                rows.append({
                    "shop_id":     shop_id,
                    "shop_name":   name,
                    "distance":    distance,
                    "medicine_id": med_id,
                    "price":       float(med_entry.get("price", 0)),
                    "quantity":    int(med_entry.get("quantity", 0)),
                })

        df = pd.DataFrame(rows)
        print(f"[Shop Optimizer] Loaded {len(df)} shop-medicine rows from {len(shops)} shops.")
        return df

    except Exception as exc:
        print(f"[Shop Optimizer] Error loading shop data: {exc}")
        return pd.DataFrame(columns=[
            "shop_id", "shop_name", "distance",
            "medicine_id", "price", "quantity",
        ])


def _build_candidates(df: pd.DataFrame) -> List[ShopCandidate]:
    return [
        ShopCandidate(
            shop_id=str(sid),
            shop_name=grp["shop_name"].iloc[0],
            distance=float(grp["distance"].iloc[0]),
            medicine_prices=dict(zip(grp["medicine_id"], grp["price"].astype(float))),
            medicine_quantities=dict(zip(grp["medicine_id"], grp["quantity"].astype(int))),
        )
        for sid, grp in df.groupby("shop_id")
    ]

def _qty_penalty(available: int, required: int) -> float:
    if required <= 0:
        return 0.0
    return max(0, required - available) / required


def _effective_price(med_id: str, c: ShopCandidate, req_qty: int) -> float:
    obtainable = min(c.medicine_quantities.get(med_id, 0), req_qty)
    return c.medicine_prices.get(med_id, 0.0) * obtainable


def _minmax(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _normalize(results: List[ShopResult]) -> List[ShopResult]:
    norms = _minmax([r.raw_score for r in results])
    for r, n in zip(results, norms):
        r.norm_score = n
    return results


def _score_candidate(
    c: ShopCandidate,
    required: Dict[str, int],
    w_dist: float,
    w_price: float,
    w_fulfil: float,
) -> Optional[ShopResult]:
    req_set = set(required)
    covered = req_set & set(c.medicine_prices)
    if not covered:
        return None

    coverage_ratio    = len(covered) / len(req_set)
    fulfillable_count = sum(
        1 for m in covered
        if c.medicine_quantities.get(m, 0) >= required.get(m, 1)
    )
    fulfillable_ratio = fulfillable_count / len(req_set)
    qty_penalty = sum(
        _qty_penalty(c.medicine_quantities.get(m, 0), required.get(m, 1))
        for m in covered
    ) / len(req_set)
    total_price = sum(_effective_price(m, c, required.get(m, 1)) for m in covered)

    raw_score = (
        w_dist  * c.distance
        + w_price * total_price
        + w_fulfil * qty_penalty * total_price
    )

    return ShopResult(
        shop_id=c.shop_id,
        shop_name=c.shop_name,
        distance=c.distance,
        covered_medicines=list(covered),
        uncovered_medicines=list(req_set - covered),
        coverage_ratio=coverage_ratio,
        fulfillable_ratio=fulfillable_ratio,
        total_price=total_price,
        raw_score=raw_score,
    )

def _greedy_cover(
    candidates: List[ShopCandidate],
    required: Dict[str, int],
    w_dist: float,
    w_price: float,
) -> List[ShopCandidate]:
    remaining = dict(required)
    chosen: List[ShopCandidate] = []

    while remaining:
        best: Optional[Tuple[float, ShopCandidate]] = None
        for c in candidates:
            new = set(remaining) & set(c.medicine_prices)
            if not new:
                continue
            marginal = (
                w_dist  * c.distance / len(new)
                + w_price * sum(_effective_price(m, c, remaining[m]) for m in new) / len(new)
            )
            if best is None or marginal < best[0]:
                best = (marginal, c)
        if not best:
            break
        chosen.append(best[1])
        for m in list(remaining):
            if m in best[1].medicine_prices:
                remaining[m] = max(0, remaining[m] - best[1].medicine_quantities.get(m, 0))
                if remaining[m] == 0:
                    del remaining[m]

    return chosen


def _eval_combo(
    combo: List[ShopCandidate],
    required: Dict[str, int],
    w_dist: float,
    w_price: float,
) -> MultiShopSolution:
    remaining   = dict(required)
    total_price = 0.0
    shop_results: List[ShopResult] = []

    for c in combo:
        new    = set(remaining) & set(c.medicine_prices)
        contrib = sum(_effective_price(m, c, remaining[m]) for m in new)
        total_price += contrib
        shop_results.append(ShopResult(
            shop_id=c.shop_id, shop_name=c.shop_name, distance=c.distance,
            covered_medicines=list(new), uncovered_medicines=[],
            coverage_ratio=len(new) / len(required),
            fulfillable_ratio=len(new) / len(required),
            total_price=contrib, raw_score=0.0,
        ))
        for m in list(remaining):
            if m in c.medicine_prices:
                remaining[m] = max(0, remaining[m] - c.medicine_quantities.get(m, 0))
                if remaining[m] == 0:
                    del remaining[m]

    max_dist = max((c.distance for c in combo), default=0.0)
    return MultiShopSolution(
        shops=shop_results,
        total_price=total_price,
        max_distance=max_dist,
        combined_score=w_dist * max_dist + w_price * total_price,
        fully_covered=len(remaining) == 0,
    )


def find_best_shops(
    required_medicine_ids: List[str],
    required_quantities: Optional[Dict[str, int]] = None,
    top_n: int = 3,
    w_distance: float = 0.6,
    w_price: float = 0.3,
    w_fulfillability: float = 0.1,
) -> List[Dict]:
    if not required_medicine_ids:
        return []

    required = {m: (required_quantities or {}).get(m, 1) for m in required_medicine_ids}
    df = _load_shop_dataframe()
    if df.empty:
        return []

    results = [
        r for r in (
            _score_candidate(c, required, w_distance, w_price, w_fulfillability)
            for c in _build_candidates(df)
        ) if r is not None
    ]
    if not results:
        return []

    _normalize(results)
    results.sort(key=lambda r: (-r.coverage_ratio, -r.fulfillable_ratio, r.norm_score))

    return [
        {
            "shop_id":             r.shop_id,
            "shop_name":           r.shop_name,
            "distance":            r.distance,
            "coverage_ratio":      round(r.coverage_ratio, 4),
            "fulfillable_ratio":   round(r.fulfillable_ratio, 4),
            "covered_medicines":   [med_info.get(mid, {}).get("name", mid) for mid in r.covered_medicines],
            "uncovered_medicines": [med_info.get(mid, {}).get("name", mid) for mid in r.uncovered_medicines],
            "total_price":         round(r.total_price, 2),
            "norm_score":          round(r.norm_score, 4),
        }
        for r in results[:top_n]
    ]


def find_best_multi_shop_solution(
    required_medicine_ids: List[str],
    required_quantities: Optional[Dict[str, int]] = None,
    max_shops: int = 3,
    top_n: int = 3,
    w_distance: float = 0.6,
    w_price: float = 0.3,
) -> List[Dict]:
    if not required_medicine_ids:
        return []

    required   = {m: (required_quantities or {}).get(m, 1) for m in required_medicine_ids}
    df         = _load_shop_dataframe()
    if df.empty:
        return []

    candidates = _build_candidates(df)
    eligible   = [c for c in candidates if set(required) & set(c.medicine_prices)]

    solutions: List[MultiShopSolution] = []

    greedy = _greedy_cover(eligible, required, w_distance, w_price)
    if greedy:
        solutions.append(_eval_combo(greedy, required, w_distance, w_price))

    pruned = sorted(eligible, key=lambda c: -len(set(required) & set(c.medicine_prices)))[:12]
    for size in range(1, min(max_shops, len(pruned)) + 1):
        for combo in combinations(pruned, size):
            solutions.append(_eval_combo(list(combo), required, w_distance, w_price))

    seen: Dict[frozenset, MultiShopSolution] = {}
    for sol in solutions:
        key = frozenset(sr.shop_id for sr in sol.shops)
        if key not in seen or sol.combined_score < seen[key].combined_score:
            seen[key] = sol

    unique = sorted(seen.values(), key=lambda s: (not s.fully_covered, s.combined_score))

med_info = {
        m["medicineId"]: {"name": m["name"], "brand": m.get("brand", "")}
        for m in _fetch_all_medicines()
    }
def enrich(med_ids):
        return [
            {
                "id":    mid,
                "name":  med_info.get(mid, {}).get("name", mid),
                "brand": med_info.get(mid, {}).get("brand", ""),
            }
            for mid in med_ids
        ]

        return [
        {
            "shop_id":             r.shop_id,
            "shop_name":           r.shop_name,
            "distance":            r.distance,
            "coverage_ratio":      round(r.coverage_ratio, 4),
            "covered_medicines":   [med_info.get(mid, {}).get("name", mid) for mid in r.covered_medicines],
            "uncovered_medicines": [med_info.get(mid, {}).get("name", mid) for mid in r.uncovered_medicines],
            "uncovered_medicines": enrich(r.uncovered_medicines),
            "total_price":         round(r.total_price, 2),
            "norm_score":          round(r.norm_score, 4),
        }
        for r in results[:top_n]
    ]
