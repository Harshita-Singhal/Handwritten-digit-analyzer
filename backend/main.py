from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

# Vercel frontend allow karne ke liye
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

# Frontend se jo data aayega uska format
class RequestData(BaseModel):
    image: str
    logic: str  # 'clean' ya 'camera'

@app.post("/predict")
async def analyze_document(data: RequestData):
    try:
        # 1. Base64 Image ko OpenCV format me convert karna
        base64_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        
        # Original image ko color me read karte hain taaki preprocessing ache se ho
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. APPLYING LOGIC (Clean Scan vs Camera Scan)
        if data.logic == "camera":
            # Camera Scan Logic: Adaptive thresholding for shadows, bad lighting, and noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
        else:
            # Clean Scan Logic (Canvas Draw or Scanner): Simple global threshold
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # 3. CONTOUR DETECTION (Multiple digits extract karna)
        # Find all numbers in the image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from Left to Right (taaki 1, 2, 3 sequence me read ho)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        if bounding_boxes:
            contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))

        extracted_digits = []
        confidences = []
        frequencies = [0] * 10  # 0 se 9 tak ke digits ka count

        # 4. PREDICT EACH DIGIT
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            
            # Filter out very small dots/noise
            if w >= 10 and h >= 10:
                # Crop the digit with some padding
                roi = thresh[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
                
                # Resize to 28x28 for the CNN Model
                try:
                    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                    roi = roi / 255.0
                    roi = roi.reshape(1, 28, 28, 1)

                    # Model Prediction
                    pred = model.predict(roi)
                    digit = int(np.argmax(pred))
                    conf = float(np.max(pred) * 100)

                    extracted_digits.append(digit)
                    confidences.append(conf)
                    frequencies[digit] += 1 # Update Graph Data
                except Exception as e:
                    continue # Skip if crop fails

        # Handle empty canvas case
        if len(extracted_digits) == 0:
            return {"error": "No digits found. Please draw or upload clearly."}

        # Calculate Overall Confidence
        avg_confidence = sum(confidences) / len(confidences)

        # 5. RETURN ADVANCED DATA TO FRONTEND
        return {
            "extracted_digits": extracted_digits,
            "frequencies": frequencies,
            "overall_confidence": round(avg_confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}
