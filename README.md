<div align="center">

# 🧠 Handwritten Digit Analyzer ✍️

A real-time, deep learning-powered web application for processing messy, distorted handwriting.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

🚀 [View Live Demo Here](https://handwritten-digit-analyzer.vercel.app/) 🚀

</div>

---

## 📌 Project Overview
Current OCR systems and standard digit classifiers often struggle with complex, unstructured documents containing background noise or grid lines. This project solves this limitation by implementing a robust data pipeline featuring HSV Color Masking and Topological Contouring to isolate valid digits. These isolated patches are then processed by a custom Convolutional Neural Network (CNN) trained on the EMNIST dataset to extract spatial features and accurately predict the final sequence.

## 📸 Project Screenshots

<div align="center">
  
| Data Pipeline & Architecture | Live Frequency Analytics |
| :---: | :---: |
| <img src="docs/architecture_diagram.png" alt="Architecture Diagram" width="400"/> | <img src="docs/frequency_graph.png" alt="Frequency Graph" width="400"/> |
| The complete data flow from raw input to predicted output. | Real-time analytics of the extracted digits. |

| Preprocessing Example | Web Interface |
| :---: | :---: |
| <img src="docs/preprocessing.png" alt="Preprocessing" width="400"/> | <img src="docs/dashboard.png" alt="Dashboard UI" width="400"/> |
| HSV masking and precise topological cropping. | The responsive frontend dashboard. |

</div>

## ✨ Key Features
* Advanced Preprocessing: Utilizes OpenCV for HSV masking (eliminating false positives and background noise) and topological mapping to crop precise bounding boxes.
* Deep Learning Engine: Features a CNN that extracts spatial features from isolated 28x28 digit patches, achieving over 98% validation accuracy.
* FastAPI Microservice: The backend is designed as a lightweight, high-performance REST API capable of live data analytics.
* Interactive Frontend: A responsive JavaScript canvas and dashboard that visualizes the data pipeline and live prediction results.

## 🏗️ System Architecture
1. Frontend (Web Interface): Captures user input via document upload or canvas and sends it as an HTTP POST request.
2. Backend (FastAPI Server): Receives the raw image data and routes it to the AI Processing Unit.
3. AI Processing Unit: 
   * Cleans the image (Noise Removal & Size Matching).
   * Feeds isolated 28x28 patches into the CNN (2D Conv & Max Pooling).
   * Passes features to Dense layers (64 Neurons) for final 0-9 classification.
   * Returns a JSON response with recognized digits.

## 🚀 Local Installation & Setup

If you wish to run the backend API and frontend locally:

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
