from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

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

class RequestData(BaseModel):
    image: str
    logic: str 

@app.post("/predict")
async def analyze_document(data: RequestData):
    try:
        # 1. Decode Image from Frontend
        base64_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        
        # Load as grayscale directly
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        # 2. THRESHOLDING
        if data.logic == "camera":
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
        else:
            # Clean Scans: Simple Binary Inverse
            _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
            
            # Dilate to connect broken lines/thin strokes
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)

        # 3. Find Outlines (Contours)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        extracted_digits = []
        confidences = []
        frequencies = [0] * 10 

        # 4. Process Contours
        valid_contours = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            # Filter out tiny dust particles (must be at least 5x5 pixels)
            if w >= 5 and h >= 5:
                valid_contours.append((x, y, w, h))

        # Sort contours Left to Right
        valid_contours = sorted(valid_contours, key=lambda b: b[0])

        for (x, y, w, h) in valid_contours:
            # Crop exactly to the digit
            roi = thresh[y:y+h, x:x+w]

            # --- SMART PADDING (Fixes the crashing and accuracy issues) ---
            # Make the image a perfect square without stretching it
            side = max(w, h)
            pad_x = (side - w) // 2
            pad_y = (side - h) // 2
            square_roi = cv2.copyMakeBorder(roi, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)

            # Add a 20% border around the square (This matches how MNIST AI was trained!)
            pad_amt = int(side * 0.2)
            final_roi = cv2.copyMakeBorder(square_roi, pad_amt, pad_amt, pad_amt, pad_amt, cv2.BORDER_CONSTANT, value=0)

            # Resize securely to 28x28
            roi_resized = cv2.resize(final_roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi_final = roi_resized / 255.0
            roi_final = roi_final.reshape(1, 28, 28, 1)

            # Predict
            pred = model.predict(roi_final)
            digit = int(np.argmax(pred))
            conf = float(np.max(pred) * 100)

            extracted_digits.append(digit)
            confidences.append(conf)
            frequencies[digit] += 1

        # 5. ULTIMATE FALLBACK (Your original logic!)
        # If the contour extraction fails entirely, just run the whole canvas
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
        # Instead of hiding the error, send the exact Python error to Vercel
        return {"error": f"Actual Backend Crash: {str(e)}"}
