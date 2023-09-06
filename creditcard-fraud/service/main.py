from typing import Annotated
import joblib
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, Query


app = FastAPI()

model: RandomForestClassifier = joblib.load("./model.bin")


@app.get("/fraud")
def predict_fraud(
    time: Annotated[float, Query(example=406)],
    v1: Annotated[float, Query(example=-2.312226542)],
    v2: Annotated[float, Query(example=1.951992011)],
    v3: Annotated[float, Query(example=-1.609850732)],
    v4: Annotated[float, Query(example=3.997905588)],
    v5: Annotated[float, Query(example=-0.522187865)],
    v6: Annotated[float, Query(example=-1.426545319)],
    v7: Annotated[float, Query(example=-2.537387306)],
    v8: Annotated[float, Query(example=1.391657248)],
    v9: Annotated[float, Query(example=-2.770089277)],
    v10: Annotated[float, Query(example=-2.772272145)],
    v11: Annotated[float, Query(example=3.202033207)],
    v12: Annotated[float, Query(example=-2.899907388)],
    v13: Annotated[float, Query(example=-0.595221881)],
    v14: Annotated[float, Query(example=-4.289253782)],
    v15: Annotated[float, Query(example=0.38972412)],
    v16: Annotated[float, Query(example=-1.14074718)],
    v17: Annotated[float, Query(example=-2.830055675)],
    v18: Annotated[float, Query(example=-0.016822468)],
    v19: Annotated[float, Query(example=0.416955705)],
    v20: Annotated[float, Query(example=0.126910559)],
    v21: Annotated[float, Query(example=0.517232371)],
    v22: Annotated[float, Query(example=-0.035049369)],
    v23: Annotated[float, Query(example=-0.465211076)],
    v24: Annotated[float, Query(example=0.320198199)],
    v25: Annotated[float, Query(example=0.044519167)],
    v26: Annotated[float, Query(example=0.177839798)],
    v27: Annotated[float, Query(example=0.261145003)],
    v28: Annotated[float, Query(example=-0.143275875)],
    amount: Annotated[float, Query(example=0)],
):
    data = [
        [
            time,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
            v24,
            v25,
            v26,
            v27,
            v28,
            amount,
        ]
    ]

    return {
        "predicted_fraud": model.predict(data).tolist()[0],
        "probability": str(model.predict_proba(data)[0]),
    }
