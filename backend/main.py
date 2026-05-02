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

# Load Model (Ensure path is correct)
try:
    model = load_model("../model/cnn_model.h5") 
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

class RequestData(BaseModel):
    image: str
    logic: str # 'clean' ya 'camera'

@app.post("/predict")
async def analyze_document(data: RequestData):
    try:
        # Decode Base64 Image
        base64_data = data.image.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # 1. Clean Preprocessing (Always convert to white-on-black binary)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if data.logic == "camera":
            # For camera scans: handle shadows, bad lighting
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
        else:
            # For clean drawing or scanner: global inversion & threshold
            # Step 1: Force true white background (255) for clean draw
            # For single drawn digit, assuming clean draw is most likely logic choice.
            thresh = cv2.bitwise_not(gray) # Assuming black ink on white background
            _, thresh = cv2.threshold(thresh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 2. Multi-Digit Detection groundwork (Keep OCR features ready)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        extracted_digits = []
        confidences = []
        frequencies = [0] * 10

        # SORT controus left-to-right to support multi-digit order in OCR tasks
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        if bounding_boxes:
            contours, bounding_boxes = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))

        # --- ADVANCED LOGIC: Robust segmentation handling ---
        
        # Determine total valid objects found (that pass size threshold)
        valid_object_count = 0
        all_objects = []

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            # Brittle size threshold for multi-digit OCR needs to be strict (10x10) to avoid noise.
            # But mandatory fragmentation for single drawn object is brittle.
            if w >= 10 and h >= 10:
                valid_object_count += 1
                roi = thresh[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
                all_objects.append(roi)
        
        # --- Handle single drawn object case ROBUSTLY ---
        
        # CASE 1: Only ONE object found, but it is smaller than threshold.
        # This is the drawn '5' case from image_1.png where Otsu or drawing thin lines caused issues.
        if valid_object_count == 0 and len(contours) == 1:
             (x, y, w, h) = cv2.boundingRect(contours[0])
             roi = thresh[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
             # Analyze this specific patch as the sole object found, ignoring multi-digit logic for single inputs.
             classify_patch(roi, extracted_digits, confidences, frequencies)

        # CASE 2: No valid objects found, and multiple small things are found. Then it is noise.
        elif valid_object_count == 0 and len(contours) > 1:
            return {"error": "No digits found. Please draw or upload clearly."}

        # CASE 3: Only ONE large object found. Handle normally.
        elif valid_object_count == 1:
            classify_patch(all_objects[0], extracted_digits, confidences, frequencies)

        # CASE 4: Multiple large objects found. Handle multi-digit/OCR features.
        elif valid_object_count > 1:
            for roi in all_objects:
                classify_patch(roi, extracted_digits, confidences, frequencies)
                
        # --- Final return with advanced features (frequencies) ---
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "extracted_digits": extracted_digits,
            "frequencies": frequencies,
            "overall_confidence": round(avg_confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}

# Helper function to classify a patch and update results
def classify_patch(roi, extracted_digits, confidences, frequencies):
    try:
        # Standard preprocessing for classification patch: resize to 28x28, normalize, reshape.
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi / 255.0
        roi = roi.reshape(1, 28, 28, 1)

        pred = model.predict(roi)
        digit = int(np.argmax(pred))
        conf = float(np.max(pred) * 100)

        extracted_digits.append(digit)
        confidences.append(conf)
        frequencies[digit] += 1
    except Exception:
        pass
