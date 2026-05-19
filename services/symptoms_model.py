from sentence_transformers import SentenceTransformer, util

from services.medicine_lookup import _fetch_all_medicines 

model = SentenceTransformer('all-MiniLM-L6-v2')

def detect_symptoms(user_input: str, top_k: int = 7) -> list:
    if not user_input or not user_input.strip():
        return []

    medicines = _fetch_all_medicines() 
    if not medicines:
        print("[Symptoms Model] No medicines found in database")
        return []

    # Filter out medicines with no uses — they add noise to similarity scores
    medicines = [m for m in medicines if m.get("uses")]
    if not medicines:
        print("[Symptoms Model] No medicines with 'uses' field found")
        return []

    medicine_uses       = [", ".join(med["uses"]) for med in medicines]
    medicine_embeddings = model.encode(medicine_uses, convert_to_tensor=True)

    print(f"[Symptoms Model] Processing input: '{user_input}'")
    query_emb = model.encode(user_input.strip(), convert_to_tensor=True)
    scores    = util.cos_sim(query_emb, medicine_embeddings)[0]

    # Cap top_k to available medicines so .topk() never throws
    top_k      = min(top_k, len(medicines))
    top_indices = scores.topk(top_k).indices.tolist()

    result = [
        {
            "id":    medicines[i]["medicineId"],
            "name":  medicines[i]["name"],
            "brand": medicines[i].get("brand", ""),
        }
        for i in top_indices
    ]

    print(f"[Symptoms Model] Found {len(result)} medicines: {[m['name'] for m in result]}")
    return result
