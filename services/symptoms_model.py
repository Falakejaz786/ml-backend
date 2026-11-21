from sentence_transformers import SentenceTransformer, util

from db import get_db

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


def detect_symptoms(user_input, top_k=7):
    """
    Detects top medicines for given symptom or user text.
    Returns top_k medicines (id, name, brand).
    """
    try:
        db = get_db()
        medicines_cursor = db.medicines.find({})
        medicines = list(medicines_cursor)

        if not medicines:
            print(f"[Symptoms Model] No medicines found in database")
            return []

    except Exception as e:
        print(f"[Symptoms Model] Error fetching medicines from MongoDB: {e}")
        return []

    if not medicines:
        return []

    # Precompute medicine uses embeddings
    medicine_uses = [", ".join(med.get("uses", [])) for med in medicines]
    medicine_embeddings = model.encode(medicine_uses, convert_to_tensor=True)

    print(f"[Symptoms Model] Processing input: '{user_input}'")
    query_emb = model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, medicine_embeddings)[0]
    top_indices = scores.topk(top_k).indices.tolist()

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
