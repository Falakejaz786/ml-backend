from typing import Dict, List, Optional, Tuple

from db import get_db


def _fetch_all_medicines() -> List[Dict]:
    
    try:
        return list(get_db().medicines.find({}))
    except Exception as exc:
        print(f"[Medicine Lookup] Error fetching medicines: {exc}")
        return []


def detect_medicine_query(text: str, max_results: int = 5) -> List[Dict]:
    """
    Find medicines matching *text* using tiered fuzzy matching.
    Returns list of dicts: {id, name, brand, score}
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return []

    medicines = _fetch_all_medicines()
    if not medicines:
        print(f"[Medicine Lookup] No medicines in DB for query: {text!r}")
        return []

    scored: List[Tuple[int, Dict]] = []

    for med in medicines:
        name_l  = med.get("name", "").lower()
        brand_l = med.get("brand", "").lower()
        uses_l  = [u.lower() for u in med.get("uses", [])]

        if med.get("medicineId", "").lower() == text_lower:
            score = 100
        elif name_l == text_lower or brand_l == text_lower:
            score = 100
        elif name_l in text_lower or text_lower in name_l:
            score = 60
        elif any(text_lower in u or u in text_lower for u in uses_l):
            score = 50
        elif brand_l and (brand_l in text_lower or text_lower in brand_l):
            score = 40
        else:
            score = 0
            for tw in text_lower.split():
                if len(tw) < 3:
                    continue
                for mw in name_l.split():
                    if len(mw) >= 3 and (mw.startswith(tw) or tw.startswith(mw)):
                        score = 25
                        break
                if score:
                    break

        if score:
            scored.append((score, med))

    scored.sort(key=lambda x: x[0], reverse=True)
    found = [
        {
            "id":    s[1]["medicineId"],
            "name":  s[1]["name"],
            "brand": s[1].get("brand", ""),
            "score": s[0],
        }
        for s in scored[:max_results]
    ]

    print(f"[Medicine Lookup] '{text}' → {[m['name'] for m in found]}")
    return found


def get_medicine_by_id(medicine_id: str) -> Optional[Dict]:
    """Fetch a single medicine document by its medicineId field."""
    try:
        return get_db().medicines.find_one({"medicineId": medicine_id})
    except Exception as exc:
        print(f"[Medicine Lookup] Error fetching medicine {medicine_id}: {exc}")
        return None
    
