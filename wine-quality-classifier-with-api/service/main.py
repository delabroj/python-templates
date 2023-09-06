from typing import Annotated
from xgboost import XGBClassifier
from fastapi import FastAPI, Query

app = FastAPI()

model = XGBClassifier()
model.load_model("model.bin")


@app.get("/wine-quality")
def predict_quality(
    typ: Annotated[int, Query(description="0 = Red, 1 = White", example=1)],
    fixed_acidity: Annotated[float, Query(example=7.3)],
    volatile_acidity: Annotated[float, Query(example=0.65)],
    citric_acid: Annotated[float, Query(example=0.0)],
    residual_sugar: Annotated[float, Query(example=1.2)],
    chlorides: Annotated[float, Query(example=0.065)],
    free_sulfur_dioxide: Annotated[float, Query(example=15)],
    density: Annotated[float, Query(example=0.9946)],
    pH: Annotated[float, Query(example=3.39)],
    sulphates: Annotated[float, Query(example=0.47)],
    alcohol: Annotated[float, Query(example=10)],
):
    data = [
        [
            typ,
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol,
        ]
    ]

    return {
        "predicted_high_quality": model.predict(data).tolist()[0],
        "probability": str(model.predict_proba(data)[0]),
    }
