from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load("our_model_Task2.joblib")

# Define the FastAPI app
# Here the server is running 
app = FastAPI()

# Define a request model


class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Define a prediction endpoint

@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Convert input data to a 2D array for the model
    input_data = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                            features.AveBedrms, features.Population, features.AveOccup,
                            features.Latitude, features.Longitude]])

    # Predict using the model
    prediction = model.predict(input_data)

    # Return the result
    # {} this means json Dictenary  
    return {"predicted_price": float(prediction[0]), "Message:": "This is the Predict " }
