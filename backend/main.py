from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

# Vercel frontend allow karne ke liye CORS zaroori hai
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
try:
    model = load_model("../model/cnn_model.h5") 
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

class CanvasData(BaseModel):
    image: str

@app.post("/predict")
async def predict_digit(data: CanvasData):
    try:
        # Decode Base64 Image
        base64_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        
        # Preprocessing
        img = cv2.bitwise_not(img) 
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predict
        predictions = model.predict(img)
        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions) * 100)

        return {
            "prediction": predicted_digit,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}
