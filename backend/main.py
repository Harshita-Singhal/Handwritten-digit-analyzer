from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SMART MODEL FINDER (Updated with Correct Name) ---
model = None

# Ab code aapki sahi file 'digit_model.h5' ko dhoondega
possible_paths = [
    "../model/digit_model.h5",   
    "model/digit_model.h5",      
    "./digit_model.h5",          
    "digit_model.h5"
]

for path in possible_paths:
    if os.path.exists(path):
        try:
            model = load_model(path)
            print(f"✅ Model loaded successfully from: {path}")
            break
        except Exception as e:
            print(f"❌ Found file at {path} but failed to load: {e}")

if model is None:
    print("🚨 CRITICAL ERROR: digit_model.h5 could not be found anywhere!")


class RequestData(BaseModel):
    image: str
    logic: str 

@app.post("/predict")
async def analyze_document(data: RequestData):
    # Agar model load nahi hua toh Render crash nahi hoga, Vercel ko error bhejega
    if model is None:
        return {"error": "AI Model (digit_model.h5) not found on Render! Check filename/path."}

    try:
        base64_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if data.logic == "camera":
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
        else:
            _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        extracted_digits = []
        confidences = []
        frequencies = [0] * 10 

        valid_contours = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w >= 5 and h >= 5:
                valid_contours.append((x, y, w, h))

        valid_contours = sorted(valid_contours, key=lambda b: b[0])

        for (x, y, w, h) in valid_contours:
            roi = thresh[y:y+h, x:x+w]
            side = max(w, h)
            pad_x = (side - w) // 2
            pad_y = (side - h) // 2
            square_roi = cv2.copyMakeBorder(roi, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)

            pad_amt = int(side * 0.2)
            final_roi = cv2.copyMakeBorder(square_roi, pad_amt, pad_amt, pad_amt, pad_amt, cv2.BORDER_CONSTANT, value=0)

            roi_resized = cv2.resize(final_roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi_final = roi_resized / 255.0
            roi_final = roi_final.reshape(1, 28, 28, 1)

            pred = model.predict(roi_final)
            digit = int(np.argmax(pred))
            conf = float(np.max(pred) * 100)

            extracted_digits.append(digit)
            confidences.append(conf)
            frequencies[digit] += 1

        if len(extracted_digits) == 0:
            roi_resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
            roi_final = roi_resized / 255.0
            roi_final = roi_final.reshape(1, 28, 28, 1)

            pred = model.predict(roi_final)
            digit = int(np.argmax(pred))
            conf = float(np.max(pred) * 100)

            extracted_digits.append(digit)
            confidences.append(conf)
            frequencies[digit] += 1

        avg_confidence = sum(confidences) / len(confidences)

        return {
            "extracted_digits": extracted_digits,
            "frequencies": frequencies,
            "overall_confidence": round(avg_confidence, 2)
        }

    except Exception as e:
        return {"error": f"Actual Backend Crash: {str(e)}"}
