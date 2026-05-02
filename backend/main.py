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
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. SMART THRESHOLDING (Bug fixed here)
        if data.logic == "camera":
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
        else:
            # Simple Binary Inverse: White bg becomes black, Black ink becomes white.
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            
            # Thicken the drawn lines so the CNN model can see them clearly
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)

        # 3. Find Outlines (Contours)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Agar canvas khali hai ya image detect nahi hui
        if len(contours) == 0:
            return {"error": "Could not detect any shapes. Please draw a thicker number."}

        # Sort contours Left to Right
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))

        extracted_digits = []
        confidences = []
        frequencies = [0] * 10 

        valid_object_count = 0
        
        # 4. Extract and Predict
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)

            # Filter out tiny dots (noise)
            if w >= 10 and h >= 10:
                valid_object_count += 1
                
                # Add padding around the digit so it isn't squeezed
                y_start = max(0, y - 10)
                y_end = min(thresh.shape[0], y + h + 10)
                x_start = max(0, x - 10)
                x_end = min(thresh.shape[1], x + w + 10)

                roi = thresh[y_start:y_end, x_start:x_end]
                classify_patch(roi, extracted_digits, confidences, frequencies)

        # Fallback: Agar sari lines 10x10 se choti thi (bohot chota draw kiya)
        if valid_object_count == 0:
            largest_c = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(largest_c)
            roi = thresh[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
            classify_patch(roi, extracted_digits, confidences, frequencies)

        # Agar sab fail ho jaye
        if len(extracted_digits) == 0:
            return {"error": "Shape found, but the AI could not process it."}

        avg_confidence = sum(confidences) / len(confidences)

        # 5. Send data back to Frontend
        return {
            "extracted_digits": extracted_digits,
            "frequencies": frequencies,
            "overall_confidence": round(avg_confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}

# Helper function for CNN prediction
def classify_patch(roi, extracted_digits, confidences, frequencies):
    try:
        if roi.size == 0:
            return

        # Resize exactly to 28x28 (what MNIST model expects)
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi / 255.0
        roi = roi.reshape(1, 28, 28, 1)

        pred = model.predict(roi)
        digit = int(np.argmax(pred))
        conf = float(np.max(pred) * 100)

        extracted_digits.append(digit)
        confidences.append(conf)
        frequencies[digit] += 1
    except Exception as e:
        print(f"Skipped patch due to error: {e}")
        pass
