"""Service to optimize shop visit plans for a user's medicine cart."""
from typing import Dict, List, Optional
from bson import ObjectId

from db import get_db


def convert_objectid(obj):
    """Recursively convert ObjectId objects to strings in a dict/list."""
    if isinstance(obj, dict):
        return {k: convert_objectid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid(v) for v in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    return obj


def optimize_visit_plan(
    medicine_ids: List[str],
    quantities: Optional[List[int]] = None,
) -> Dict:
    """Generate an optimized visit plan to collect all medicines in the cart.

    Uses a greedy algorithm: repeatedly pick the nearest shop that covers 
    the most remaining unassigned medicines until all are assigned or unavailable.

    Args:
        medicine_ids: List of medicine IDs in the cart
        quantities: List of quantities (unused for now, but can be extended)

    Returns:
        Dict with:
        - stops: List of shops to visit in order with medicines to collect there
        - unavailable: List of medicines that can't be found in any shop
    """

    if not medicine_ids:
        return {"stops": [], "unavailable": []}

    db = get_db()

    # Fetch all shop-medicine relationships
    shop_medicines = list(db.shopmedicines.find({}))
    shops_collection = list(db.shops.find({}))
    medicines_collection = list(db.medicines.find({}))

    # Build lookup maps
    shops_by_id = {shop["shopId"]: shop for shop in shops_collection}
    medicines_by_id = {med["medicineId"]: med for med in medicines_collection}

    # Build shop medicine availability map: {shopId: {medicineId: {price, quantity}}}
    shop_med_map: Dict[str, Dict[str, dict]] = {}
    for sm in shop_medicines:
        shop_obj_id = str(sm.get("shop"))
        shop = next(
            (s for s in shops_collection if str(s["_id"]) == shop_obj_id),
            None
        )
        if not shop:
            continue

        shop_id = shop.get("shopId")
        med_obj_id = str(sm.get("medicine"))
        medicine = next(
            (m for m in medicines_collection if str(m["_id"]) == med_obj_id),
            None
        )
        if not medicine:
            continue

        med_id = medicine.get("medicineId")

        if shop_id not in shop_med_map:
            shop_med_map[shop_id] = {}

        shop_med_map[shop_id][med_id] = {
            "price": sm.get("price", 0),
            "quantity": sm.get("quantity", 0),
        }

    # Find medicines unavailable anywhere
    unavailable = [
        med_id for med_id in medicine_ids
        if not any(med_id in shop_med_map.get(sid, {})
                   for sid in shop_med_map)
    ]

    # Build cart quantities map: medicine_id -> quantity_needed
    medicine_quantities: Dict[str, int] = {}
    if quantities:
        for med_id, qty in zip(medicine_ids, quantities):
            medicine_quantities[med_id] = qty
    else:
        # Default to 1 if not specified
        medicine_quantities = {med_id: 1 for med_id in medicine_ids}

    remaining_meds = set(medicine_ids) - set(unavailable)
    stops: List[Dict] = []

    while remaining_meds:
        best_shop_id: Optional[str] = None
        best_covered = set()
        best_distance = float('inf')

        # Score each shop by how many remaining medicines it covers
        for shop_id, med_map in shop_med_map.items():
            if shop_id not in shops_by_id:
                continue

            covered = remaining_meds & set(med_map.keys())
            if not covered:
                continue

            shop = shops_by_id[shop_id]
            # Parse distance like "1.2 km" -> 1.2 or handle if already numeric
            distance_value = shop.get("distance_from_user", "9999 km")
            try:
                # Handle both float and string types
                if isinstance(distance_value, (int, float)):
                    distance = float(distance_value)
                else:
                    distance = float(distance_value.split()[0])
            except (ValueError, IndexError, AttributeError):
                distance = 9999.0

            # Pick shop with most coverage; tiebreak by nearest
            is_better = (
                len(covered) > len(best_covered) or
                (len(covered) == len(best_covered) and distance < best_distance)
            )

            if is_better:
                best_shop_id = shop_id
                best_covered = covered
                best_distance = distance

        if not best_shop_id:
            break  # No more shops can cover remaining medicines

        stop = shops_by_id[best_shop_id]
        medicines_at_shop = [
            {
                "medicine_id": med_id,
                "name": medicines_by_id.get(med_id, {}).get("name", med_id),
                "brand": medicines_by_id.get(med_id, {}).get("brand", ""),
                "form": medicines_by_id.get(med_id, {}).get("form", ""),
                "price": shop_med_map[best_shop_id][med_id].get("price", 0),
                "quantity": shop_med_map[best_shop_id][med_id].get("quantity", 0),
            }
            for med_id in sorted(best_covered)
        ]

        # Fetch owner's name if owner is a user ID
        owner_field = stop.get("owner", "")
        owner_name = str(owner_field) if owner_field else ""

        # Check if owner field looks like an ID (try to find in users collection)
        # Convert to string first to check length
        owner_str = str(owner_field) if owner_field else ""
        if owner_field and len(owner_str) > 20:  # Likely an ObjectId or long ID
            try:
                # Try looking up by userId
                owner_user = db.users.find_one({"userId": owner_str})
                if owner_user and owner_user.get("name"):
                    owner_name = owner_user["name"]
                else:
                    # Try looking up by _id if it's 24 chars (ObjectId)
                    if len(owner_str) == 24:
                        try:
                            owner_user = db.users.find_one(
                                {"_id": ObjectId(owner_str)})
                            if owner_user and owner_user.get("name"):
                                owner_name = owner_user["name"]
                        except:
                            pass
            except:
                pass  # Keep owner_name as fallback

        stops.append({
            "shop_id": best_shop_id,
            "shop_name": stop.get("name", ""),
            "owner": owner_name,
            "owner_name": owner_name,
            "phone": stop.get("phone", ""),
            "location": stop.get("location", ""),
            "distance": stop.get("distance_from_user", "0 km"),
            "distance_km": best_distance,
            "medicines": medicines_at_shop,
            "medicine_count": len(medicines_at_shop),
            "total_price": sum(
                shop_med_map[best_shop_id][med_id]["price"] *
                medicine_quantities.get(med_id, 1)
                for med_id in best_covered
            ),
        })

        remaining_meds -= best_covered

    # Sort stops by distance (nearest first)
    stops.sort(key=lambda s: s["distance_km"])

    result = {
        "stops": stops,
        "unavailable": [
            {
                "medicine_id": med_id,
                "name": medicines_by_id.get(med_id, {}).get("name", med_id),
            }
            for med_id in unavailable
        ],
        "total_medicines": len(medicine_ids),
        "available_medicines": len(medicine_ids) - len(unavailable),
        "total_shops": len(stops),
        "total_cost": sum(stop["total_price"] for stop in stops),
    }

    return convert_objectid(result)
