from db import get_db

model = None
medicine_embeddings = None
cached_medicine_signature = None
cached_medicines = None


def _keyword_fallback(user_input, medicines, top_k):
    words = set(user_input.lower().replace(",", " ").replace("-", " ").split())
    ranked = []

    for medicine in medicines:
        uses = " ".join(str(use) for use in medicine.get("uses", [])).lower()
        name = medicine.get("name", "").lower()
        score = sum(1 for word in words if len(word) >= 3 and word in uses)
        score += sum(1 for word in words if len(word) >= 3 and word in name)
        if score:
            ranked.append((score, medicine))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "id": medicine["medicineId"],
            "name": medicine["name"],
            "brand": medicine.get("brand", ""),
        }
        for _, medicine in ranked[:top_k]
    ]


def detect_symptoms(user_input, top_k=7):
    """
    Detects top medicines for given symptom or user text.
    Returns top_k medicines (id, name, brand).
    """
    global model, medicine_embeddings, cached_medicine_signature, cached_medicines

    try:
        if cached_medicines is not None:
            medicines = cached_medicines
        else:
            db = get_db()
            medicines_cursor = db.medicines.find({})
            medicines = list(medicines_cursor)
            cached_medicines = medicines

        if not medicines:
            print(f"[Symptoms Model] No medicines found in database")
            return []

    except Exception as e:
        print(f"[Symptoms Model] Error fetching medicines from MongoDB: {e}")
        return []

    if not medicines:
        return []

    medicine_signature = tuple(
        (med.get("medicineId", ""), tuple(med.get("uses", [])))
        for med in medicines
    )

    if model is None:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[Symptoms Model] Embedding model unavailable: {e}")
            return _keyword_fallback(user_input, medicines, top_k)

    from sentence_transformers import util

    if medicine_embeddings is None or cached_medicine_signature != medicine_signature:
        try:
            medicine_uses = [", ".join(med.get("uses", [])) for med in medicines]
            medicine_embeddings = model.encode(
                medicine_uses,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            cached_medicine_signature = medicine_signature
            cached_medicines = medicines
        except Exception as e:
            print(f"[Symptoms Model] Embedding failed: {e}")
            return _keyword_fallback(user_input, medicines, top_k)
    else:
        medicines = cached_medicines

    print(f"[Symptoms Model] Processing input: '{user_input}'")
    try:
        query_emb = model.encode(
            user_input,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        scores = util.cos_sim(query_emb, medicine_embeddings)[0]
        top_indices = scores.topk(min(top_k, len(medicines))).indices.tolist()
    except Exception as e:
        print(f"[Symptoms Model] Embedding query failed: {e}")
        return _keyword_fallback(user_input, medicines, top_k)

    result = []
    for idx in top_indices:
        med = medicines[idx]
        result.append({
            "id": med["medicineId"],
            "name": med["name"],
            "brand": med.get("brand", "")
        })

    print(
        f"[Symptoms Model] Found {len(result)} medicines: {[m['name'] for m in result]}")
    return result
