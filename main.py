from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.medicine_lookup import detect_medicine_query
from services.shop_optimizer import find_best_shops
from services.symptoms_model import detect_symptoms
from services.visit_plan_optimizer import optimize_visit_plan


app = FastAPI(title="Medical Symptom Checker API")

# Add CORS middleware to handle preflight requests
app.add_middleware(
    CORSMiddleware,
    # Allow all origins (can be restricted to specific domains)
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)


class SymptomsRequest(BaseModel):
    symptoms: str = Field(...,
                          description="User symptoms in natural language")
    top_shops: int | None = Field(
        None,
        description="(Optional) number of top shops to return. Defaults to 3.",
    )


class MedicinesRequest(BaseModel):
    medicines: list[str] = Field(...,
                                 description="List of medicine IDs or medicine names")
    top_shops: int | None = Field(
        None,
        description="(Optional) number of top shops to return. Defaults to 5.",
    )


class CartItem(BaseModel):
    medicine_id: str = Field(..., description="Medicine ID")
    quantity: int = Field(default=1, description="Quantity needed")


class OptimizeCartRequest(BaseModel):
    cart_items: list[CartItem] = Field(
        ...,
        description="List of medicine IDs and quantities in the user's cart"
    )


@app.post("/symptoms")
def get_medicines_for_symptoms_endpoint(request: SymptomsRequest):
    """Takes natural language symptoms and returns matching medicines with best shops.

    Uses semantic embedding to match symptoms with medicine uses.
    Returns medicines ranked by relevance + best shops where they can be found.
    """

    symptoms_text = (request.symptoms or "").strip()
    if not symptoms_text:
        raise HTTPException(
            status_code=400, detail="`symptoms` must be non-empty")

    # 1) Use ML model to detect relevant medicines for symptoms
    matched_medicines = detect_symptoms(symptoms_text)

    if not matched_medicines:
        return {
            "query_type": "symptoms",
            "query": symptoms_text,
            "matched_medicines": [],
            "best_shops": [],
        }

    # 2) Find best shops that carry these medicines
    med_ids = [m["id"] for m in matched_medicines]
    shops = find_best_shops(med_ids, top_n=request.top_shops or 3)

    return {
        "query_type": "symptoms",
        "query": symptoms_text,
        "matched_medicines": matched_medicines,
        "best_shops": shops,
    }


@app.post("/medicines")
def find_shops_for_medicines_endpoint(request: MedicinesRequest):
    """Takes a list of medicines and returns best shops that carry all of them.

    For each medicine identifier (name or ID), finds matching medicines.
    Returns the best shops ranked by:
    - Coverage (shops that carry all requested medicines)
    - Distance (shortest distance)
    - Price (lowest total price)
    """

    if not request.medicines:
        raise HTTPException(
            status_code=400, detail="`medicines` list must not be empty")

    # 1) Resolve all medicine identifiers to IDs
    all_medicine_ids = set()
    resolved_medicines = []
    unresolved_medicines = []

    print(f"[Medicines Endpoint] Searching for: {request.medicines}")

    for med_identifier in request.medicines:
        med_identifier = med_identifier.strip()
        if not med_identifier:
            continue

        # Try to match by name first
        matches = detect_medicine_query(med_identifier, max_results=1)
        print(
            f"[Medicines Endpoint] Query '{med_identifier}': {len(matches)} matches")
        if matches:
            all_medicine_ids.add(matches[0]["id"])
            resolved_medicines.extend(matches)
        else:
            # Couldn't find this medicine - track it for response
            unresolved_medicines.append(med_identifier)
            # Still add it as an ID in case it's already an ID format
            all_medicine_ids.add(med_identifier)

    # 2) If we couldn't resolve any medicines, return helpful error
    if not resolved_medicines and unresolved_medicines:
        print(f"[Medicines Endpoint] No medicines resolved")
        return {
            "query_type": "medicines",
            "requested_medicines": unresolved_medicines,
            "matched_medicines": [],
            "best_shops": [],
        }

    # 3) Find best shops that carry these medicines
    med_ids_list = list(all_medicine_ids)
    print(
        f"[Medicines Endpoint] Finding shops for {len(med_ids_list)} medicines: {med_ids_list}")
    shops = find_best_shops(med_ids_list, top_n=request.top_shops or 5)
    print(f"[Medicines Endpoint] Found {len(shops)} shops")

    return {
        "query_type": "medicines",
        "requested_medicines": med_ids_list,
        "matched_medicines": resolved_medicines,
        "best_shops": shops,
    }


@app.post("/chat")
def chat(request: SymptomsRequest | MedicinesRequest):
    """Legacy endpoint - Respond to a user query.

    If the message includes a known medicine name, returns the best shops for that medicine.
    Otherwise, attempts to interpret the message as symptoms and returns medicines + shops.

    DEPRECATED: Use /symptoms or /medicines endpoints instead.
    """

    if isinstance(request, MedicinesRequest):
        message = "".join(request.medicines)
    else:
        message = request.symptoms

    message = (message or "").strip()
    if not message:
        raise HTTPException(
            status_code=400, detail="message must be non-empty")

    # 1) Check if the user asked for a specific medicine by name.
    medicine_matches = detect_medicine_query(message, max_results=5)
    if medicine_matches:
        med_ids = [m["id"] for m in medicine_matches]
        shops = find_best_shops(med_ids, top_n=5)

        return {
            "query_type": "medicine",
            "query": message,
            "matched_medicines": medicine_matches,
            "best_shops": shops,
        }

    # 2) Otherwise, treat input as symptom description.
    matched_medicines = detect_symptoms(message)
    med_ids = [m["id"] for m in matched_medicines]
    shops = find_best_shops(med_ids, top_n=3)

    return {
        "query_type": "symptoms",
        "query": message,
        "matched_medicines": matched_medicines,
        "best_shops": shops,
    }


@app.post("/optimize-cart")
def optimize_cart_endpoint(request: OptimizeCartRequest):
    """Generates an optimized visit plan to collect all medicines in the user's cart.

    Takes a list of medicines and their quantities, then uses a greedy algorithm
    to determine the best shops to visit in order (prioritized by coverage and distance).

    Returns shops sorted by nearest first, with the medicines to collect at each shop.
    """

    if not request.cart_items:
        raise HTTPException(
            status_code=400, detail="cart_items must not be empty")

    medicine_ids = [item.medicine_id for item in request.cart_items]
    quantities = [item.quantity for item in request.cart_items]

    print(
        f"[Optimize Cart] Optimizing visit plan for {len(medicine_ids)} medicines: {medicine_ids}")

    visit_plan = optimize_visit_plan(medicine_ids, quantities)

    print(
        f"[Optimize Cart] Generated plan with {len(visit_plan['stops'])} stops")

    return visit_plan


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
