from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

# This allows your Vercel website to talk to this server securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your custom-trained AI Brain
print("Loading model...")
model = tf.keras.models.load_model('digit_model.h5')
print("Model loaded successfully!")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the image sent from the website
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Use OpenCV to find the digits
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Make background black and ink white
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Draw invisible boxes around anything that looks like a number
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    
    # Get the total size of the uploaded image
    img_height, img_width = thresh.shape
    # Set a maximum area (ignore anything larger than 20% of the total image)
    max_area = (img_height * img_width) * 0.2 
    
    # 3. Process each box one by one
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        # FILTER 1: Ignore tiny dots (dust)
        if area < 60:
            continue
            
        # FILTER 2: Ignore massive boxes (like the application window borders)
        if area > max_area:
            continue
            
        # FILTER 3: Ignore thin lines (a real digit has some width and height)
        if h < 15 or w < 5:
            continue
            
        # Crop the image to just the number, adding a 15px border
        digit_img = thresh[max(0, y-15):y+h+15, max(0, x-15):x+w+15]
        
        if digit_img.size == 0:
            continue
            
        # Resize to 28x28 pixels because that's what our AI expects
        resized = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 28, 28, 1))
        
        # 4. Ask the AI to predict the number!
        prediction = model.predict(reshaped)
        predicted_digit = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        # Save the result
        results.append({
            "digit": int(predicted_digit),
            "confidence": confidence,
            "box": {"x": x, "y": y, "w": w, "h": h}
        })

    # Send the answers back to the website
    return {"status": "success", "total_found": len(results), "data": results}
