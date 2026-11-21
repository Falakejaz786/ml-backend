
from typing import Dict, List, Optional

from db import get_db


def detect_medicine_query(text: str, max_results: int = 5) -> List[Dict]:
    """Find medicines matching the user query using flexible matching.

    Tries multiple matching strategies:
    1. Exact match (case-insensitive): text.lower() == med["name"].lower()
    2. Partial match: medicine name is substring of text or text is substring of medicine name
    3. Word-based match: meaningful word matching (min 3 chars) on medicine name
    """

    text_lower = text.lower().strip()
    found = []
    scores = []  # Track match quality

    try:
        db = get_db()
        medicines_cursor = db.medicines.find({})
        medicines = list(medicines_cursor)

        if not medicines:
            print(
                f"[Medicine Lookup] No medicines found in database for query: {text}")
            return []

    except Exception as e:
        print(f"[Medicine Lookup] Error fetching medicines from MongoDB: {e}")
        return []

    for med in medicines:
        med_name_lower = med.get("name", "").lower()
        med_brand_lower = med.get("brand", "").lower()
        med_uses = [use.lower() for use in med.get("uses", [])]

        score = 0

        # Strategy 1: Exact match on name or brand (highest priority)
        if med_name_lower == text_lower or med_brand_lower == text_lower:
            score = 100
        # Strategy 2: Case-insensitive substring match on name
        elif med_name_lower in text_lower or text_lower in med_name_lower:
            score = 60
        # Strategy 3: Check if text matches medicine uses
        elif any(text_lower in use or use in text_lower for use in med_uses):
            score = 50
        # Strategy 4: Brand name substring match
        elif med_brand_lower and (med_brand_lower in text_lower or text_lower in med_brand_lower):
            score = 40
        # Strategy 5: Meaningful word-based matching (require min 3 chars)
        else:
            text_words = text_lower.split()
            med_words = med_name_lower.split()

            # Only match if word is at least 3 characters to avoid single-letter matches
            for text_word in text_words:
                if len(text_word) < 3:
                    continue

                for med_word in med_words:
                    # Require meaningful prefix match (at least 3 chars shared)
                    if len(med_word) >= 3 and med_word.startswith(text_word):
                        score = 25
                        break
                    elif len(text_word) >= len(med_word) and text_word.startswith(med_word) and len(med_word) >= 3:
                        score = 25
                        break
                if score > 0:
                    break

        if score > 0:
            scores.append((score, med))

        if len(scores) >= max_results * 2:  # Get extra results to sort
            break

    # Sort by score (descending) and take top results
    if scores:
        scores.sort(key=lambda x: x[0], reverse=True)
        # Return the properly formatted dicts
        found = [
            {
                "id": scores[i][1]["medicineId"],
                "name": scores[i][1]["name"],
                "brand": scores[i][1].get("brand", ""),
            }
            for i in range(min(max_results, len(scores)))
        ]
        print(
            f"[Medicine Lookup] Found {len(found)} medicines for '{text}': {[m['name'] for m in found]}")
    else:
        print(f"[Medicine Lookup] No matches found for '{text}'")

    return found


def get_medicine_by_id(medicine_id: str) -> Optional[Dict]:
    try:
        db = get_db()
        med = db.medicines.find_one({"medicineId": medicine_id})
        return med
    except Exception as e:
        print(f"Error fetching medicine from MongoDB: {e}")
        return None
