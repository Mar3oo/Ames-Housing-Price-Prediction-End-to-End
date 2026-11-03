from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from housing.predict import load_model, predict_price


app = FastAPI(
    title="Ames Housing Price Prediction API",
    description="API for estimating residential property sale prices.",
    version="1.0.0",
)

# Load the model once when the API starts
model = load_model()


class PredictionRequest(BaseModel):
    features: dict[str, Any]


class BatchPredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)


class PredictionResponse(BaseModel):
    predicted_price: float


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    total_properties: int


@app.get("/")
def root():
    return {
        "message": "Ames Housing Price Prediction API",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/schema")
def get_model_schema():
    return {
        "required_features": model.feature_names_in_.tolist(),
        "total_features": len(model.feature_names_in_),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        input_data = pd.DataFrame([request.features])
        prediction = predict_price(input_data=input_data, model=model)

        return PredictionResponse(predicted_price=float(prediction.iloc[0]))

    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(request: BatchPredictionRequest):
    try:
        input_data = pd.DataFrame(request.records)

        predictions = predict_price(
            input_data=input_data,
            model=model,
        )

        return BatchPredictionResponse(
            predictions=[float(value) for value in predictions],
            total_properties=len(predictions),
        )

    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
