
import time
import uuid

from contextlib import asynccontextmanager

import pandas as pd

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_loading import ModelLoader
from api.preprocessing import clean_prediction_data

from api.schemas import (
    EmployeeInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

from api.monitor_logging import log_inference


# Create the model loader object.
model_loader = ModelLoader()


# Application startup and shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup: load model and preprocessor.
    model_loader.load()

    yield

    # Shutdown.
    print("[INFO] Employee Attrition API shutting down...")


# Create FastAPI application.
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Lab 10 - ML Model Serving and API Engineering",
    version="1.0.0",
    lifespan=lifespan
)


# Handle invalid input data.
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):

    request_id = uuid.uuid4().hex[:12]

    errors = []

    for error in exc.errors():

        locations = [
            str(location)
            for location in error["loc"]
            if location != "body"
        ]

        field = ".".join(locations)

        errors.append({
            "field": field,
            "message": error["msg"],
        })

    # Log validation failure.
    log_inference(
        input_data={},
        latency_ms=None,
        status="failure",
        error="Request validation failed",
        model_name=model_loader.model_name,
        model_version=model_loader.model_version,
        run_id=model_loader.run_id,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Request validation failed",
            "details": errors,
        },
    )


# Handle HTTP exceptions.
@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": "Prediction service error",
            "details": [
                {
                    "field": "prediction",
                    "message": str(exc.detail),
                }
            ],
        },
    )


# Root endpoint.
@app.get("/")
def root():

    return {
        "message": "Employee Attrition Prediction API is running",
        "lab": "Lab 10",
        "status": "success",
    }


# Health endpoint.
@app.get("/health")
def health():

    ready = model_loader.is_ready()

    return {
        "status": "healthy" if ready else "not_ready",
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }


# Model metadata endpoint.
@app.get("/metadata")
def metadata():

    return {
        "service": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "model_name": model_loader.model_name,
        "model_uri": model_loader.model_uri,
        "model_version": model_loader.model_version,
        "run_id": model_loader.run_id,
        "model_loaded": model_loader.model is not None,
        "preprocessor_loaded": model_loader.preprocessor is not None,
    }


# Single employee prediction endpoint.
@app.post("/predict", response_model=PredictionResponse)
def predict(data: EmployeeInput):

    request_id = uuid.uuid4().hex[:12]

    start_time = time.perf_counter()

    data_dict = data.model_dump()

    # Check model availability.
    if not model_loader.is_ready():

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3
        )

        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error="Model or preprocessor is not available",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
            include_input=True
        )

        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:

        # 1. Convert JSON input into a DataFrame.
        df = pd.DataFrame([data_dict])

        # 2. Clean the raw input.
        df, employee_numbers = clean_prediction_data(df)

        # 3. Transform using the saved preprocessor.
        X = model_loader.preprocessor.transform(df)

        # 4. Generate prediction.
        prediction = model_loader.model.predict(X)[0]

        # 5. Generate attrition probability.
        probability = model_loader.model.predict_proba(X)[0][1]

        # 6. Calculate inference latency.
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3
        )

        # 7. Log inference event.
        log_inference(
            input_data=data_dict,
            prediction=int(prediction),
            attrition_probability=float(probability),
            latency_ms=latency_ms,
            status="success",
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
            include_input=True
        )

        # 8. Return prediction.
        return {
            "prediction": int(prediction),
            "attrition_probability": float(probability),
        }

    except Exception as e:

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3
        )
        import logging

        logging.exception("Prediction endpoint failed")

        log_inference(
            input_data=data_dict,
            latency_ms=latency_ms,
            status="failure",
            error=str(e),
            model_name=model_loader.model_name,
            model_version=model_loader.model_version,
            run_id=model_loader.run_id,
            request_id=request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction",
        )


# Batch employee prediction endpoint.
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse
)
def predict_batch(data: BatchPredictionRequest):

    start_time = time.perf_counter()

    # Convert employee objects into dictionaries.
    data_dicts = [
        employee.model_dump()
        for employee in data.employees
    ]

    # Check model availability.
    if not model_loader.is_ready():

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3
        )

        for data_dict in data_dicts:

            log_inference(
                input_data=data_dict,
                latency_ms=latency_ms,
                status="failure",
                error="Model or preprocessor is not available",
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=uuid.uuid4().hex[:12],
                include_input=True
            )

        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor is not available",
        )

    try:

        # 1. Convert all employees into a DataFrame.
        df = pd.DataFrame(data_dicts)

        # 2. Clean incoming employee data.
        df = clean_prediction_data(df)

        # 3. Transform using saved preprocessor.
        X = model_loader.preprocessor.transform(df)

        # 4. Generate predictions for all employees.
        predictions = model_loader.model.predict(X)

        # 5. Generate attrition probabilities.
        probabilities = model_loader.model.predict_proba(X)[:, 1]

        # 6. Calculate total batch latency.
        total_latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3
        )

        results = []

        # 7. Log each employee's prediction.
        for data_dict, prediction, probability in zip(
            data_dicts,
            predictions,
            probabilities,
        ):

            request_id = uuid.uuid4().hex[:12]

            log_inference(
                input_data=data_dict,
                prediction=int(prediction),
                attrition_probability=float(probability),
                latency_ms=total_latency_ms,
                status="success",
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=request_id,
                include_input=True
            )

            results.append({
                "prediction": int(prediction),
                "attrition_probability": float(probability),
            })

        # 8. Return batch results.
        return {
            "count": len(results),
            "predictions": results,
        }

    except Exception as e:

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            3
        )

        for data_dict in data_dicts:

            log_inference(
                input_data=data_dict,
                latency_ms=latency_ms,
                status="failure",
                error=str(e),
                model_name=model_loader.model_name,
                model_version=model_loader.model_version,
                run_id=model_loader.run_id,
                request_id=uuid.uuid4().hex[:12],
                include_input=True
            )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during batch prediction",
        )
