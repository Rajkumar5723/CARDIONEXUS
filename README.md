
# 🫀 CARDIONEXUS: AI-Powered Heart Disease Diagnostic Platform

CARDIONEXUS is a smart healthcare web application built using **Flask** that enables users to upload **ECG images**, **scanned health reports**, and manually input symptoms to detect heart conditions using machine learning and deep learning models. The system integrates **OCR** (Optical Character Recognition), **image processing**, and trained models to provide accurate diagnosis support.

---

## 🔍 Key Features

- 🧠 **AI-Powered Diagnosis** from ECG and Reports
- 📄 **OCR Parsing** for Extracting Text from Scanned Medical Reports
- 📈 **Prediction of Heart Conditions** such as:
  - Myocardial Infarction (MI)
  - Hypertension
  - Arrhythmia, and others
- 👤 **Login and Signup** Authentication System
- 🧾 **Report Upload, Viewing, and Debugging**
- 📊 Dashboard and Statistics Visualization

---

## 🏗️ Tech Stack

| Layer        | Tools & Frameworks                              |
|--------------|--------------------------------------------------|
| Backend      | Flask, Python                                    |
| Frontend     | HTML5, CSS3, JavaScript                          |
| ML/DL Models | TensorFlow, Keras, scikit-learn, joblib          |
| OCR Engine   | Tesseract OCR                                    |
| Image Utils  | PIL, OpenCV, skimage                             |
| Database     | MongoDB                                          |

---

## 📁 Folder Structure

```
CARDIONEXUS/
├── app.py
├── templates/                 # HTML Templates
│   ├── login.html
│   ├── dashboard.html
│   └── ...
├── static/                    # CSS, JS, Image Files
├── models/                   # ML/DL models (Pickle/Keras)
│   ├── ecg_model.h5
│   ├── report_model.pkl
│   └── ...
├── utils/                     # Helper modules for OCR and prediction
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Rajkumar5723/CARDIONEXUS.git
cd CARDIONEXUS
```

### 2. Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Tesseract Path (Windows Only)

In `app.py`, update the path:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Download: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

### 5. Run the Application

```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## 🧪 Sample Inputs

- **ECG Image**: Upload `.png`, `.jpg`, `.jpeg` images of ECG waveforms.
- **Medical Report**: Scanned lab reports (OCR extractable).
- **Manual Input**: Enter symptoms if reports/images unavailable.

---

## 🤖 Model Overview

- **ECG Classifier**: Deep CNN (trained on waveform datasets)  
- **OCR Extractor**: Tesseract OCR + Preprocessing Pipeline  
- **Text Classifier**: TF-IDF + Logistic Regression / SVM  
- **LLM**: Fine-tuned `CardioMed-LLaMA3.2-1B` model used for awareness and treatment content generation  
- **Ensemble Output**: Based on confidence scores and data type

---

## 📌 Future Scope

- Real-time integration with hospital equipment
- Patient history & EMR integration
- Cross-language support for regional health centers
- Mobile App version with camera-based ECG capture
- Explainable AI (XAI) features

---

## 🛡️ Security

- Session-based user authentication
- Encrypted model loading and prediction logic
- MongoDB user database with role control

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

**Rajkumar G**

- 📧 Email: g.p.rajkumar5@gmail.com  
- 🌐 GitHub: [https://github.com/Rajkumar5723](https://github.com/Rajkumar5723)  
- 🌐 LinkedIn: [https://linkedin.com/in/rajkumar](https://linkedin.com/in/rajkumar)
- 🌐 Portfolio: [https://rajkumarg.vercel.app/](https://rajkumarg.vercel.app/)
  

---

> Made with ❤️ for AI in Healthcare 🚑
