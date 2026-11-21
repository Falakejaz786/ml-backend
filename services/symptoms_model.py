import threading

from db import get_db

model = None
medicine_embeddings = None
cached_medicine_signature = None
cached_medicines = None
model_lock = threading.Lock()
model_loaded = threading.Event()
model_error = None


def load_symptom_model():
    """Load the embedding model and medicine embeddings before handling requests."""
    global model, medicine_embeddings, cached_medicine_signature
    global cached_medicines, model_error

    if model_loaded.is_set():
        return True

    with model_lock:
        if model_loaded.is_set():
            return True

        try:
            if model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-MiniLM-L6-v2")

            if cached_medicines is None:
                db = get_db()
                cached_medicines = list(db.medicines.find({}))

            medicine_signature = tuple(
                (medicine.get("medicineId", ""), tuple(medicine.get("uses", [])))
                for medicine in cached_medicines
            )

            if (
                medicine_embeddings is None
                or cached_medicine_signature != medicine_signature
            ):
                medicine_uses = [
                    ", ".join(medicine.get("uses", []))
                    for medicine in cached_medicines
                ]
                medicine_embeddings = model.encode(
                    medicine_uses,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                cached_medicine_signature = medicine_signature

            model_error = None
            model_loaded.set()
            print("[Symptoms Model] Embedding model and medicines are ready")
            return True
        except Exception as e:
            model_error = str(e)
            print(f"[Symptoms Model] Background model loading failed: {e}")
            return False


def detect_symptoms(user_input, top_k=7):
    """Detect top medicines for symptom text using cached embeddings."""
    global cached_medicines

    if not model_loaded.is_set():
        print("[Symptoms Model] Waiting for background model initialization")
        if not model_loaded.wait(timeout=120):
            print(
                "[Symptoms Model] Model initialization did not finish "
                f"within 120 seconds: {model_error}"
            )
            return []

    from sentence_transformers import util

    if not cached_medicines:
        return []

    print(f"[Symptoms Model] Processing input: '{user_input}'")
    query_emb = model.encode(
        user_input,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    scores = util.cos_sim(query_emb, medicine_embeddings)[0]
    top_indices = scores.topk(min(top_k, len(cached_medicines))).indices.tolist()

    result = [
        {
            "id": cached_medicines[index]["medicineId"],
            "name": cached_medicines[index]["name"],
            "brand": cached_medicines[index].get("brand", ""),
        }
        for index in top_indices
    ]

    print(
        f"[Symptoms Model] Found {len(result)} medicines: "
        f"{[medicine['name'] for medicine in result]}"
    )
    return result
