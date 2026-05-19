from typing import Dict, List, Optional

from bson import ObjectId

from db import get_db
from services.medicine_lookup import _fetch_all_medicines
from services.shop_optimizer import _build_candidates, _load_shop_dataframe, _greedy_cover


def convert_objectid(obj):
    if isinstance(obj, dict):
        return {k: convert_objectid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid(v) for v in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    return obj


def _resolve_owner_name(db, owner_field) -> str:
    if not owner_field:
        return ""
    owner_str = str(owner_field)
    try:
        user = db.users.find_one({"userId": owner_str})
        if user and user.get("name"):
            return user["name"]
        if len(owner_str) == 24:
            user = db.users.find_one({"_id": ObjectId(owner_str)})
            if user and user.get("name"):
                return user["name"]
    except Exception:
        pass
    return owner_str


def optimize_visit_plan(medicine_ids: List[str], quantities: Optional[List[int]] = None,) -> Dict:
    
    if not medicine_ids:
        return {"stops": [], "unavailable": []}

    medicine_quantities: Dict[str, int] = (
        dict(zip(medicine_ids, quantities))
        if quantities
        else {mid: 1 for mid in medicine_ids}
    )

    df = _load_shop_dataframe()
    if df.empty:
        return {"stops": [], "unavailable": [
            {"medicine_id": mid, "name": mid} for mid in medicine_ids
        ]}

    candidates      = _build_candidates(df)
    medicines_by_id = {m["medicineId"]: m for m in _fetch_all_medicines()}
    db              = get_db()
    shops_meta      = {s.get("shopId"): s for s in db.shops.find({})}

    all_stocked = set()
    for c in candidates:
        all_stocked.update(c.medicine_prices.keys())

    unavailable = [mid for mid in medicine_ids if mid not in all_stocked]
    required    = {
        mid: medicine_quantities[mid]
        for mid in medicine_ids
        if mid not in unavailable
    }

    if not required:
        return {
            "stops": [],
            "unavailable": [
                {"medicine_id": mid, "name": medicines_by_id.get(mid, {}).get("name", mid)}
                for mid in unavailable
            ],
            "total_medicines":    len(medicine_ids),
            "available_medicines": 0,
            "total_shops":        0,
            "total_cost":         0.0,
        }

    chosen = _greedy_cover(candidates, required, w_dist=0.7, w_price=0.3)

    stops: List[Dict] = []
    remaining = set(required.keys())

    for c in chosen:
        covered = remaining & set(c.medicine_prices.keys())
        if not covered:
            continue

        meta       = shops_meta.get(c.shop_id, {})
        owner_name = _resolve_owner_name(db, meta.get("owner", ""))

        medicines_at_stop = [
            {
                "medicine_id": mid,
                "name":        medicines_by_id.get(mid, {}).get("name", mid),
                "brand":       medicines_by_id.get(mid, {}).get("brand", ""),
                "form":        medicines_by_id.get(mid, {}).get("form", ""),
                "price":       c.medicine_prices[mid],
                "quantity":    c.medicine_quantities.get(mid, 0),
            }
            for mid in sorted(covered)
        ]

        stop_total = sum(
            c.medicine_prices[mid] * medicine_quantities.get(mid, 1)
            for mid in covered
        )

        stops.append({
            "shop_id":        c.shop_id,
            "shop_name":      c.shop_name,
            "owner":          owner_name,
            "owner_name":     owner_name,
            "phone":          meta.get("phone", ""),
            "location":       meta.get("location", ""),
            "distance":       meta.get("distance_from_user", 0),
            "distance_km":    c.distance,
            "medicines":      medicines_at_stop,
            "medicine_count": len(medicines_at_stop),
            "total_price":    stop_total,
        })

        remaining -= covered

    stops.sort(key=lambda s: s["distance_km"])

    result = {
        "stops":                stops,
        "unavailable": [
            {
                "medicine_id": mid,
                "name":        medicines_by_id.get(mid, {}).get("name", mid),
            }
            for mid in unavailable
        ],
        "total_medicines":      len(medicine_ids),
        "available_medicines":  len(medicine_ids) - len(unavailable),
        "total_shops":          len(stops),
        "total_cost":           sum(s["total_price"] for s in stops),
    }

    return convert_objectid(result)
