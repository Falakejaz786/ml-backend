from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.medicine_lookup import detect_medicine_query
from services.shop_optimizer import find_best_shops, find_best_multi_shop_solution
from services.symptoms_model import detect_symptoms
from services.visit_plan_optimizer import optimize_visit_plan


app = FastAPI(title="Medley API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymptomsRequest(BaseModel):
    symptoms: str = Field(..., description="User symptoms in natural language")
    top_shops: int | None = Field(None, description="Number of top shops to return (default 3)")


class MedicinesRequest(BaseModel):
    medicines: list[str] = Field(..., description="List of medicine IDs or names")
    top_shops: int | None = Field(None, description="Number of top shops to return (default 5)")
    quantities: dict[str, int] | None = Field(
        None,
        description="Optional {medicine_id: quantity} map. Defaults to 1 each.",
    )


class CartItem(BaseModel):
    medicine_id: str = Field(..., description="Medicine ID")
    quantity: int = Field(default=1, description="Quantity needed")


class OptimizeCartRequest(BaseModel):
    cart_items: list[CartItem] = Field(..., description="Medicines and quantities in the cart")
    multi_shop: bool = Field(
        default=False,
        description="If true, also return the best multi-shop combination plan",
    )


@app.post("/symptoms")
def symptoms_endpoint(request: SymptomsRequest):
    """
    Natural language symptoms → matched medicines → best shops.
    Uses semantic embedding (sentence-transformers) for medicine matching.
    """
    symptoms_text = (request.symptoms or "").strip()
    if not symptoms_text:
        raise HTTPException(status_code=400, detail="`symptoms` must be non-empty")

    matched_medicines = detect_symptoms(symptoms_text)
    if not matched_medicines:
        return {
            "query_type": "symptoms",
            "query": symptoms_text,
            "matched_medicines": [],
            "best_shops": [],
        }

    med_ids = [m["id"] for m in matched_medicines]
    shops   = find_best_shops(med_ids, top_n=request.top_shops or 3)

    return {
        "query_type":        "symptoms",
        "query":             symptoms_text,
        "matched_medicines": matched_medicines,
        "best_shops":        shops,
    }


@app.post("/medicines")
def medicines_endpoint(request: MedicinesRequest):
    """
    List of medicine names/IDs → resolved IDs → best shops (single + multi-shop).
    Unresolved names are tracked and returned so the client can surface them.
    """
    if not request.medicines:
        raise HTTPException(status_code=400, detail="`medicines` list must not be empty")

    resolved_medicines  = []
    unresolved_medicines = []
    all_medicine_ids     = set()

    for identifier in request.medicines:
        identifier = identifier.strip()
        if not identifier:
            continue
        matches = detect_medicine_query(identifier, max_results=1)
        if matches:
            all_medicine_ids.add(matches[0]["id"])
            resolved_medicines.extend(matches)
        else:
            unresolved_medicines.append(identifier)
            all_medicine_ids.add(identifier)   # pass through in case it's already an ID

    if not resolved_medicines and unresolved_medicines:
        return {
            "query_type":          "medicines",
            "requested_medicines": unresolved_medicines,
            "matched_medicines":   [],
            "best_shops":          [],
            "multi_shop_solutions": [],
        }

    med_ids_list = list(all_medicine_ids)
    top_n        = request.top_shops or 5

    shops = find_best_shops(
        med_ids_list,
        required_quantities=request.quantities,
        top_n=top_n,
    )
    multi = find_best_multi_shop_solution(
        med_ids_list,
        required_quantities=request.quantities,
        top_n=top_n,
    )

    return {
        "query_type":           "medicines",
        "requested_medicines":  med_ids_list,
        "matched_medicines":    resolved_medicines,
        "unresolved_medicines": unresolved_medicines,
        "best_shops":           shops,
        "multi_shop_solutions": multi,
    }


@app.post("/optimize-cart")
def optimize_cart_endpoint(request: OptimizeCartRequest):
    """
    Cart items → optimized visit plan (nearest-first, greedy coverage).
    Optionally also returns the best multi-shop combination via shop_optimizer.
    """
    if not request.cart_items:
        raise HTTPException(status_code=400, detail="`cart_items` must not be empty")

    medicine_ids = [item.medicine_id for item in request.cart_items]
    quantities   = [item.quantity    for item in request.cart_items]
    qty_map      = dict(zip(medicine_ids, quantities))

    visit_plan = optimize_visit_plan(medicine_ids, quantities)

    # Optionally attach the multi-shop optimizer result for comparison
    if request.multi_shop:
        visit_plan["multi_shop_solutions"] = find_best_multi_shop_solution(
            medicine_ids,
            required_quantities=qty_map,
            top_n=3,
        )

    return visit_plan


@app.post("/chat")
def chat_endpoint(request: SymptomsRequest | MedicinesRequest):
    """
    Legacy smart-routing endpoint.
    Tries medicine name match first; falls back to symptom detection.

    DEPRECATED: prefer /symptoms or /medicines.
    """
    if isinstance(request, MedicinesRequest):
        message = " ".join(request.medicines)
    else:
        message = request.symptoms

    message = (message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="`message` must be non-empty")

    medicine_matches = detect_medicine_query(message, max_results=5)
    if medicine_matches:
        med_ids = [m["id"] for m in medicine_matches]
        return {
            "query_type":        "medicine",
            "query":             message,
            "matched_medicines": medicine_matches,
            "best_shops":        find_best_shops(med_ids, top_n=5),
        }

    matched_medicines = detect_symptoms(message)
    med_ids = [m["id"] for m in matched_medicines]
    return {
        "query_type":        "symptoms",
        "query":             message,
        "matched_medicines": matched_medicines,
        "best_shops":        find_best_shops(med_ids, top_n=3),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
