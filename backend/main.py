from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model = tf.keras.models.load_model('digit_model.h5')
print("Model loaded successfully!")

# Added "filter_mode" to receive the dropdown choice from the frontend
@app.post("/predict")
async def predict(file: UploadFile = File(...), filter_mode: str = Form("balanced")):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- DUAL PREPROCESSING SWITCH ---
    if filter_mode == "advanced":
        # MODE 1: Camera Photo (Handles shadows and blur, but distorts thin digital lines)
        sharpen_kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
        blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    else:
        # MODE 2: Digital Scan / Default (Perfect for clean documents, fails on shadows)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- SMART EXTRACTION ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_height, img_width = thresh.shape
    max_area = (img_height * img_width) * 0.2 
    
    results = []
    
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        # FILTERS: Ignore dust, giant borders, and thin lines
        if area < 60: continue
        if area > max_area: continue
        if h < 15 or w < 5: continue
            
        digit_img = thresh[max(0, y-15):y+h+15, max(0, x-15):x+w+15]
        if digit_img.size == 0: continue
            
        # --- AI PREDICTION ---
        resized = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 28, 28, 1))
        
        prediction = model.predict(reshaped)
        predicted_digit = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        results.append({
            "digit": int(predicted_digit),
            "confidence": confidence,
            "box": {"x": x, "y": y, "w": w, "h": h}
        })

    return {"status": "success", "total_found": len(results), "data": results}
