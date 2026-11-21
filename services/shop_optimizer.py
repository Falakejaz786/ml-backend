from typing import Dict, List, Optional

import pandas as pd

from db import get_db


def _load_shop_dataframe() -> pd.DataFrame:
    """Load shop-medicine data from MongoDB into a pandas DataFrame."""
    # Always fetch fresh data from MongoDB (no caching to avoid stale data)
    try:
        db = get_db()

        # Fetch all shop-medicine relationships from correct collection name
        shop_medicines = list(db.shopmedicines.find({}))
        print(
            f"[Shop Optimizer] Found {len(shop_medicines)} shop-medicine relationships")

        if not shop_medicines:
            print("[Shop Optimizer] No shop-medicine data found in database")
            return pd.DataFrame(columns=[
                "shop_id", "shop_name", "distance", "medicine_id", "price", "quantity"
            ])

        # Fetch shops to get their names and distances
        shops_list = list(db.shops.find({}))
        shops = {str(shop["_id"]): shop for shop in shops_list}
        print(f"[Shop Optimizer] Found {len(shops)} shops")

        # Fetch medicines to map ObjectId to medicineId
        medicines_list = list(db.medicines.find({}))
        medicines = {str(med["_id"]): med for med in medicines_list}
        print(f"[Shop Optimizer] Found {len(medicines)} medicines")

        rows: List[Dict] = []

        for shop_med in shop_medicines:
            shop_id = str(shop_med.get("shop"))
            med_obj_id = str(shop_med.get("medicine"))

            shop = shops.get(shop_id)
            medicine = medicines.get(med_obj_id)

            if not shop or not medicine:
                continue

            rows.append({
                "shop_id": shop.get("shopId", shop_id),
                "shop_name": shop.get("name", ""),
                "distance": shop.get("distance_from_user", 0.0),
                # Use medicineId for matching
                "medicine_id": medicine.get("medicineId", med_obj_id),
                "price": float(shop_med.get("price", 0)),
                "quantity": int(shop_med.get("quantity", 0)),
            })

        df = pd.DataFrame(rows)
        print(f"[Shop Optimizer] Loaded dataframe with {len(df)} rows")
        return df

    except Exception as e:
        print(f"[Shop Optimizer] Error loading shop data from MongoDB: {e}")
        return pd.DataFrame(columns=[
            "shop_id", "shop_name", "distance", "medicine_id", "price", "quantity"
        ])


def find_best_shops(
    required_medicine_ids: List[str],
    top_n: int = 3,
    w_distance: float = 0.7,
    w_price: float = 0.3,
) -> List[Dict]:
    """Return the top shops that best satisfy the requested medicines.

    If no single shop carries all required medicines, we still return shops that
    carry the most medicines (highest coverage), ranked by (coverage, score).

    Score is computed as a weighted sum of distance and total price.
    """

    if not required_medicine_ids:
        return []

    df = _load_shop_dataframe()

    if df.empty:
        return []

    required_set = set(required_medicine_ids)
    best_shops: List[Dict] = []

    for shop_id, group in df.groupby("shop_id"):
        meds = set(group["medicine_id"])
        covered = required_set & meds
        if not covered:
            continue

        total_price = float(
            group[group["medicine_id"].isin(covered)]["price"].sum())
        distance = float(group["distance"].iloc[0])
        coverage = len(covered) / len(required_set)

        score = w_distance * distance + w_price * total_price

        best_shops.append(
            {
                "shop_id": shop_id,
                "shop_name": group["shop_name"].iloc[0],
                "distance": distance,
                "coverage": coverage,
                "covered_count": len(covered),
                "required_count": len(required_set),
                "total_price": total_price,
                "score": score,
            }
        )

    best_shops.sort(key=lambda s: (-s["coverage"], s["score"]))
    return best_shops[:top_n]
