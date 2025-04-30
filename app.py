from flask import Flask, request, render_template, redirect, url_for, flash,session,jsonify
from pymongo import MongoClient
from PIL import Image
import pytesseract
import random
import pickle
import re
import io
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage import color
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from skimage import measure
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging to suppress warnings
logging.basicConfig(level=logging.ERROR)

app = Flask(__name__)
app.secret_key = 'cts-project-heart-pred'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
"""
home 
login
signup
symptom
user-db creation
dashboard*
report-debug*

"""
class ECG:
    def getImage(self, image_stream):
        image = imread(image_stream)
        return image

    def GrayImage(self, image):
       
        image_gray = color.rgb2gray(image)
        image_gray = resize(image_gray, (1572, 2213))
        return image_gray

    def DividingLeads(self, image):
        
        Lead_1 = image[300:600, 150:643]  # Lead 1
        Lead_2 = image[300:600, 646:1135] # Lead aVR
        Lead_3 = image[300:600, 1140:1625] # Lead V1
        Lead_4 = image[300:600, 1630:2125] # Lead V4
        Lead_5 = image[600:900, 150:643]   # Lead 2
        Lead_6 = image[600:900, 646:1135]  # Lead aVL
        Lead_7 = image[600:900, 1140:1625] # Lead V2
        Lead_8 = image[600:900, 1630:2125] # Lead V5
        Lead_9 = image[900:1200, 150:643]  # Lead 3
        Lead_10 = image[900:1200, 646:1135] # Lead aVF
        Lead_11 = image[900:1200, 1140:1625] # Lead V3
        Lead_12 = image[900:1200, 1630:2125] # Lead V6
        Lead_13 = image[1250:1480, 150:2125] # Long Lead

        Leads = [Lead_1, Lead_2, Lead_3, Lead_4, Lead_5, Lead_6, Lead_7, Lead_8, Lead_9, Lead_10, Lead_11, Lead_12, Lead_13]
        return Leads

    def PreprocessingLeads(self, Leads):
        
        preprocessed_leads = []
        for y in Leads[:len(Leads) - 1]:
            grayscale = color.rgb2gray(y)
            blurred_image = gaussian(grayscale, sigma=1)
            global_thresh = threshold_otsu(blurred_image)
            binary_global = blurred_image < global_thresh
            binary_global = resize(binary_global, (300, 450))
            preprocessed_leads.append(binary_global)
        
        # Preprocess the last lead separately
        grayscale = color.rgb2gray(Leads[-1])
        blurred_image = gaussian(grayscale, sigma=1)
        global_thresh = threshold_otsu(blurred_image)
        binary_global = blurred_image < global_thresh
        preprocessed_leads.append(binary_global)
        
        return preprocessed_leads

    def SignalExtraction_Scaling(self, Leads):
        
        scaler = MinMaxScaler()
        all_scaled_signals = []

        for x, y in enumerate(Leads[:len(Leads) - 1]):
            grayscale = color.rgb2gray(y)
            blurred_image = gaussian(grayscale, sigma=0.7)
            global_thresh = threshold_otsu(blurred_image)
            binary_global = blurred_image < global_thresh
            binary_global = resize(binary_global, (300, 450))
            contours = measure.find_contours(binary_global, 0.8)
            contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
            
            for contour in contours:
                if contour.shape in contours_shape:
                    test = resize(contour, (255, 2))
            
            fit_transform_data = scaler.fit_transform(test)
            Normalized_Scaled = pd.DataFrame(fit_transform_data[:, 0], columns=['X'])
            Normalized_Scaled = Normalized_Scaled.T
            all_scaled_signals.append(Normalized_Scaled)

        return all_scaled_signals

    def CombineConvert1Dsignal(self, all_scaled_signals):
        
        test_final = pd.concat(all_scaled_signals, axis=1, ignore_index=True)
        return test_final

    def DimensionalReduction(self, test_final):
        
        pca_loaded_model = joblib.load("models/PCA_ECG.pkl")
        result = pca_loaded_model.transform(test_final)
        final_df = pd.DataFrame(result)
        return final_df

    def ModelLoad_predict(self, final_df):
        
        loaded_model = joblib.load("models/final.pkl")
        result = loaded_model.predict(final_df)
        
        # session['ecg_pred_conf']=max(loaded_model.predict_proba(final_df))
        if result[0] == 2:
            return "Normal"
        else:
            return "Myocardial Infarction"


with open(r'models/disease_type_classifier.pkl', 'rb') as model_file:
    disease_type_classifier = pickle.load(model_file)

with open(r'models/symptoms_classifier.pkl', 'rb') as symptoms_file:
    symptoms_classifier = pickle.load(symptoms_file)

with open(r'models/symptoms_classifier_label.pkl', 'rb') as encoder_file:
    disease_label = pickle.load(encoder_file)

hrt_atk_model = load_model('models\Myocardial_Infarction_detection_model.h5')
hrt_atk_scaler = joblib.load('models\scaler_Myocardial_Infarction.pkl')
aor_aneu_model = load_model('models\Aortic_aneurysm_detection_model.h5')
aor_aneu_scaler = joblib.load('models\scaler_aortic_aneurysm.pkl')
hyper_ten_model = load_model('models\hypertension_detection_model.h5')
hyper_ten_scaler = joblib.load('models\scaler_hypertension.pkl')
stroke_model = load_model('models\stroke_detection_model.h5')
stroke_scaler = joblib.load('models\scaler_stoke.pkl')

def get_db_connection():
    try:
        client = MongoClient("mongodb+srv://RK:96299@cluster0.gsjxsww.mongodb.net/?retryWrites=true&w=majority")
        db = client['cts']  
        print("Connected to MongoDB Atlas successfully!")
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB Atlas: {e}")
        return None
    
def encode_age(age):
    if 1 <= age <= 5:
        return 1
    elif 6 <= age <= 20:
        return 2
    elif 21 <= age <= 50:
        return 3
    elif age >= 51:
        return 4
    else:
        return 0  # Default or invalid age handling

def get_patient_details():
    patient = {
        'P_id': session.get('P_id', 'None'),
        'P_Name': session.get('P_Name', 'None'),
        'P_Age': session.get('P_age', 'None'),
        'P_PhoneNo': session.get('P_PhoneNo', 'None'),
        'report_type': session.get('report_type', 'None'),
        'syp_pred': session.get('syp_pred', 'None'),
        # 'syp_pred_conf': session.get('syp_pred_conf', 'None'),
        'report_pred': session.get('report_pred', 'None'),
        # 'report_pred_conf': session.get('report_pred_conf', 'None'),
        'ecg_pred': session.get('ecg_pred', 'None'),
        # 'ecg_pred_conf': session.get('ecg_pred_conf', 'None'),
        'act_symptoms': session.get('act_symptoms', 'None')
    }
    # for key, value in patient.items():
    #     print(f"{key}: {type(value)}")
    # print(patient)
    # patient={

    # }
    return patient

def check_hrt_atk_values(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)

    patterns = {
        'Troponin_I' : r'Troponin_\|\s+([\d.]+)',
        'Troponin_T' :r'Troponin_T\s+([\d.]+)',
        'SEX': r'Sex\s*:\s*(\w+)'
    }

    
    extracted_values = {}

    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_values[key] = match.group(1) if key in ['SEX'] else float(match.group(1))
        else:
            extracted_values[key] = None
    # Extract relevant values
    Troponin_I = extracted_values.get('Troponin_I', 0.0)
    Troponin_T = extracted_values.get('Troponin_T', 0.0)
    gender = 1 if extracted_values.get('SEX', '').lower() == 'male' else 0


    
    manual_input = np.array([[gender,Troponin_I,Troponin_T]])

    
    manual_input_scaled = hrt_atk_scaler.transform(manual_input)

    
    prediction = hrt_atk_model.predict(manual_input_scaled)


    result = "Heart Attack" if prediction[0] > 0.5 else "Healthy"
    # session['report_pred']=result
    # session['report_pred_conf']=prediction[0] *100
    # print(result)
    return result

def check_aortic_aneurysm_values(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    patterns = {
        'Systolic_BP': r'Systolic_BP\s+(\d+\.?\d*)',
        'Diastolic_BP': r'Diastolic_BP\s+(\d+\.?\d*)',
        'HDL Cholesterol': r'HDL Cholesterol\s+(\d+\.?\d*)',
        'LDL Cholesterol': r'LDL Cholesterol\s+(\d+\.?\d*)',
        'Triglycerides': r'Triglycerides\s+(\d+\.?\d*)',
        'SEX': r'Sex\s*:\s*(\w+)'
    }

    # Initialize a dictionary to store the results
    extracted_values = {}

    # Search for each pattern in the extracted text
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_values[key] = match.group(1) if key in ['SEX'] else float(match.group(1))
        else:
            extracted_values[key] = None

    # Extract relevant values
    Systolic_BP = extracted_values.get('Systolic_BP', 0.0)
    Diastolic_BP = extracted_values.get('Diastolic_BP', 0.0)
    hdl_cholesterol = extracted_values.get('HDL Cholesterol', 0.0)        
    ldl_Cholesterol = extracted_values.get('LDL Cholesterol', 0.0)        
    Triglycerides = extracted_values.get('Triglycerides', 0.0)        

    gender = 1 if extracted_values.get('SEX', '').lower() == 'male' else 0

    # Create the input array
    manual_input = np.array([[Systolic_BP,Diastolic_BP,ldl_Cholesterol,hdl_cholesterol,Triglycerides,gender]])

    # Normalize the input using the loaded scaler
    manual_input_scaled = aor_aneu_scaler.transform(manual_input)

    # Predict using the loaded model
    prediction = aor_aneu_model.predict(manual_input_scaled)

    # Interpreting the prediction
    result = "aortic aneurysm Disease" if prediction[0] > 0.5 else "Healthy"
    # session['report_pred']=result
    # session['report_pred_conf']=prediction[0]*100
    return result

def check_hypertension_values(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    patterns = {
        'Systolic_BP': r'Systolic_BP\s+([\d.]+)',
        'Diastolic_BP': r'Diastolic_BP\s+([\d.]+)',
        'SEX': r'Sex\s*:\s*(\w+)'
    }
    # Initialize a dictionary to store the results
    extracted_values = {}

    # Search for each pattern in the extracted text
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_values[key] = match.group(1) if key in ['SEX'] else float(match.group(1))
        else:
            extracted_values[key] = None
    Systolic_BP = extracted_values.get('Systolic_BP', 0.0)
    Diastolic_BP = extracted_values.get('Diastolic_BP', 0.0)
    gender = 1 if extracted_values.get('SEX', '').lower() == 'male' else 0


    # Create the input array
    manual_input = np.array([[gender,Systolic_BP,Diastolic_BP]])

    # Normalize the input using the loaded scaler
    manual_input_scaled = hyper_ten_scaler.transform(manual_input)

    # Predict using the loaded model
    prediction = hyper_ten_model.predict(manual_input_scaled)

    # Interpreting the prediction
    result = "Stroke Disease Present" if prediction[0] > 0.5 else "Healthy"
    # session['report_pred']=result
    # session['report_pred_conf']=prediction[0]*100
    return result

def check_stroke_values(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    patterns = {
        'HDL Cholesterol': r'HDL Cholesterol\s+(\d+\.?\d*)',
        'LDL Cholesterol': r'LDL Cholesterol\s+(\d+\.?\d*)',
        'Glucose Fasting': r'GLUCOSE, FASTING,\s+(\d+\.?\d*)',
        'SEX': r'Sex\s*:\s*(\w+)'
    }

    # Initialize a dictionary to store the results
    extracted_values = {}

    # Search for each pattern in the extracted text
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_values[key] = match.group(1) if key in ['SEX'] else float(match.group(1))
        else:
            extracted_values[key] = None
    
    # Extract relevant values
    hdl_cholesterol = extracted_values.get('HDL Cholesterol', 0.0)
    ldl_cholesterol = extracted_values.get('LDL Cholesterol', 0.0)
    fasting_blood_sugar = extracted_values.get('Glucose Fasting', 0.0)
    gender = 1 if extracted_values.get('SEX', '').lower() == 'male' else 0

    systolic_bp = float(request.form['systolic_bp'])
    diastolic_bp = float(request.form['diastolic_bp'])

    # Create the input array
    manual_input = np.array([[systolic_bp, diastolic_bp, hdl_cholesterol, ldl_cholesterol, fasting_blood_sugar, gender]])

    # Normalize the input using the loaded scaler
    manual_input_scaled = stroke_scaler.transform(manual_input)

    # Predict using the loaded model
    prediction = stroke_model.predict(manual_input_scaled)

    # Interpreting the prediction
    result = "Stroke Disease Present" if prediction[0] > 0.5 else "Healthy"
    # session['report_pred']=result
    # session['report_pred_conf']=prediction[0]*100
    return result

db = get_db_connection()
isLogin = False

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/login.html')
def render_login():    
    return render_template('login.html')

@app.route('/signup.html')
def render_signup():
    return render_template('signup.html')

@app.route('/signup',methods=['POST'])
def signup():
    role = request.form.get('role')
    username = request.form.get('username')
    email  = request.form.get('email')
    password = request.form.get('password')
    user ={
        'username': username,
        'email':email,
        'password':password
    }

    if db is not None:
        try:
            users_collection = db['labtech']
            if role=='doctor':
                users_collection = db['doctor'] 
            users_collection.insert_one(user)
            flash('signup successful')
            return redirect('/login.html')
        except Exception as e:
            print(e)
            return redirect('/signup.html')
    return redirect('/signup.html')

@app.route('/doctor.html') #lab-tech incomplete
def doctorPage():
    if isLogin:
        return render_template('/doctor.html',username=session.get('username'),email=session.get('email'))
    else:
        # return render_template('/doc4.html',username=username,email=email)
        return render_template('/login.html')

@app.route('/symptom.html',methods=['POST'])
def load_symptomPage():
    global isLogin
    if isLogin:
        PName = request.form.get('patientName')
        PPhoneNo= request.form.get('patientPhoneNo')
        if db is not None:
            try:
                patient_collection = db['patient']
                patient = patient_collection.find_one({'P_Name':PName,'P_PhoneNo':PPhoneNo})

                if patient is None:
                    P_id =random.randint(10000,99999)
                    patient ={
                        'P_id': P_id,
                        'P_Name': PName,
                        'P_PhoneNo':PPhoneNo,
                    }
                    patient_collection.insert_one(patient)
                    
                session['P_id']=patient['P_id']
                session['P_Name']=patient['P_Name']
                session['P_PhoneNo']=patient['P_PhoneNo'] 
                     
                return render_template('/symptom.html',P_id=session.get('P_id'),P_Name=session.get('P_Name'))  
            except Exception as e:
                print(e)
                print('error')
                
        return redirect('/signup.html')
    else:
        return redirect('/login.html')

@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    if not isLogin:
        return redirect('/login.html')

    # 1) build the symptom vector
    symptoms_order = [
        'Irritability', 'Heart Block/Failure', 'Rash', 'Feeding', 'Crying', 
        'Shortness of Breath', 'Heart Beat', 'Chest Pain', 'Fatigue', 'Sweating',
        'Dizziness', 'Nausea', 'Pain Radiation', 'Fainting', 'Coughing up blood',
        'Coughing up (without blood)', 'Swelling', 'Cyanosis', 'Weight Loss',
        'Fever', 'Bloating', 'Headache', 'Nosebleeding', 'Seizure',
        'Wheezing', 'Bodyache', 'Breathlessness'
    ]
    symptoms_array = [0] * len(symptoms_order)

    # 2) age encoding
    raw_age = int(request.form.get('age', 0))
    session['P_age'] = raw_age
    age = encode_age(raw_age)

    # 3) vital symptoms
    for symptom in ['Heart Block/Failure','Shortness of Breath','Chest Pain','Fatigue','Dizziness']:
        symptoms_array[symptoms_order.index(symptom)] = int(request.form.get(symptom, 0))

    # 4) any other free-text symptoms
    other_text = request.form.get('otherSymtoms', '')
    acting = [s for s in ['Heart Block/Failure','Shortness of Breath','Chest Pain','Fatigue','Dizziness']
              if symptoms_array[symptoms_order.index(s)] == 1]
    for i, keyword in enumerate(symptoms_order):
        if keyword not in acting and re.search(keyword, other_text, re.IGNORECASE):
            symptoms_array[i] = 1
            acting.append(keyword)

    session['act_symptoms'] = acting

    # 5) isolate the features your disease_type model expects
    isolated = [
        age,
        symptoms_array[symptoms_order.index('Irritability')],
        symptoms_array[symptoms_order.index('Feeding')],
        symptoms_array[symptoms_order.index('Shortness of Breath')],
        symptoms_array[symptoms_order.index('Heart Beat')],
        symptoms_array[symptoms_order.index('Chest Pain')],
        symptoms_array[symptoms_order.index('Fatigue')],
        symptoms_array[symptoms_order.index('Sweating')],
        symptoms_array[symptoms_order.index('Dizziness')],
        symptoms_array[symptoms_order.index('Nausea')],
        symptoms_array[symptoms_order.index('Pain Radiation')],
        symptoms_array[symptoms_order.index('Fainting')],
        symptoms_array[symptoms_order.index('Coughing up (without blood)')],
        symptoms_array[symptoms_order.index('Swelling')],
        symptoms_array[symptoms_order.index('Cyanosis')],
        symptoms_array[symptoms_order.index('Fever')],
        symptoms_array[symptoms_order.index('Wheezing')],
        symptoms_array[symptoms_order.index('Bodyache')]
    ]

    # 6) predict disease type & then final disease label
    disease_type = disease_type_classifier.predict([isolated])[0]
    features = [age] + symptoms_array + [disease_type]
    idx = symptoms_classifier.predict([features])[0]
    disease_name = disease_label.inverse_transform([idx])[0]
    session['syp_pred'] = disease_name

    # 7) generate LLM content and render
    patient = get_patient_details()
    level1, level4 = generate_content(disease_name)
    return render_template(
        'dashboard.html',
        patient=patient,
        level1=level1,
        level4=level4,
        disease=disease_name
    )
    

@app.route('/clear')
def clear_ses():
    session.clear()
    return redirect('/')
    
@app.route('/labtechnician.html')
def render_labtechnician():
    if isLogin:
        return render_template('/labtechnician.html',username=session.get('username'),email=session.get('email'))
    else:
        return render_template('/login.html')
    
@app.route('/login', methods=['POST'])
def login():
    global isLogin
    # Get data from the submitted form
    role = request.form.get('role')
    username = request.form.get('username')
    password = request.form.get('password')
    # print(role+" "+username+" "+password)
    
    if db is not None:
        try:
            users_collection = db['labtech']
            if role=='doctor':
                users_collection = db['doctor'] 
             # Replace 'users' with your collection name
            user = users_collection.find_one({'username': username, 'password': password})
            print(user)
            if user:
                # print('login ss')
                session['username']=user['username']
                session['email']=user['email']
                
                # print(user)
                # If a matching user is found, login is successful
                flash('Login successful!', 'success')
                # Redirect to a dashboard or home page
                isLogin= True
                if(role=='doctor'):
                    # return render_template('/doctor.html',username=session.get('username'),email=session.get('email'))
                    return redirect('/doctor.html')
                else:
                    # return render_template('/labtechnician.html',username=session.get('username'),email=session.get('email'))
                    return redirect('/labtechnician.html')
                
            else:
                # If no matching user is found, login fails
                # print('login failed')
                flash('Invalid username or password. Please try again.', 'error')
                return redirect(url_for('home'))
        except Exception as e:
            print(f"Error querying MongoDB Atlas: {e}")
            # flash('An error occurred. Please try again later.', 'error')
            # return redirect(url_for('home'))

    # Redirect to the home page after processing
    return redirect(url_for('home')) 
  
@app.route('/predict_report', methods=['POST'])
def predict_report():
    # Get data from the form
    session['P_id']=request.form.get('patient-id')
    session['P_Name']=request.form.get('patientName')
    session['report_type'] = request.form.getlist('report')[0]
    return render_template('/labtech_sub.html')
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------- LLM Integration --------
llm_name = "Rajkumar57/CardioMed-LLaMA3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
generator = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_unique_content(prompt: str, max_length: int = 512) -> str:
    """
    Generate unique medical content from the fine-tuned LLaMA model.
    Splits on '### Response:' if present.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(generator.device)
    outputs = generator.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("new text generated")
    # extract after response marker if available
    return full_text.split("### Response:")[-1].strip()


def generate_content(disease: str) -> tuple[str, str]:
    """
    Generate two pieces of content: abstract & awareness (level1) and
    treatments & complications (level4) for the specified disease.
    Returns a tuple (level1_content, level4_content).
    """
    prompt_1 = (
        f"### Instruction:\n"
        f"Provide an abstract and awareness information for the following disease: {disease}\n"
        f"\n### Response:\n"
    )
    prompt_4 = (
        f"### Instruction:\n"
        f"Create awareness about treatments and complications of {disease}.\n"
        f"\n### Response:\n"
    )
    level1 = generate_unique_content(prompt_1)
    level4 = generate_unique_content(prompt_4)
    print("name")
    return level1, level4

@app.route('/uploadReport', methods=['POST'])
def process_report():
    if 'testFile' not in request.files:
        return redirect('/lab_tec_sub.html')

    image_file = request.files['testFile']
    
    # Determine the disease based on report type
    if session['report_type'] == 'ar':
        disease = 'Aortic Aneurysm'
        result = check_aortic_aneurysm_values(image_file)
    elif session['report_type'] == 'hrt-atk':
        disease = 'Myocardial Infarction'
        result = check_hrt_atk_values(image_file)
    elif session['report_type'] == 'hyp-ten':
        disease = 'Hypertension'
        result = check_hypertension_values(image_file)
    elif session['report_type'] == 'stroke':
        disease = 'Stroke'
        result = check_stroke_values(image_file)

    # Generate content for the disease
    level1, level4 = generate_content(disease)

    # Store the report prediction in the session
    session['report_pred'] = result

    # Process ECG file if present
    if 'ecgFile' in request.files:
        file = request.files['ecgFile']
        if file:
            image_stream = BytesIO(file.read())
            ecg = ECG()
            ecg_user_image = ecg.getImage(image_stream)
            ecg_user_gray_image = ecg.GrayImage(ecg_user_image)
            dividing_leads = ecg.DividingLeads(ecg_user_image)
            ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
            ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
            ecg_1dsignal = ecg.CombineConvert1Dsignal(ec_signal_extraction)
            ecg_final = ecg.DimensionalReduction(ecg_1dsignal)
            ecgmodel = ecg.ModelLoad_predict(ecg_final)
            session['ecg_pred'] = ecgmodel

    # Get patient details and render the template with the generated content
    patient = get_patient_details()
    return render_template('dashboard.html', patient=patient, level1=level1, level4=level4,disease=disease)

    # Extract values from the image

# @app.route('/uploadECG', methods=['POST'])  
# def process_ecg():
#     if 'testFile' not in request.files:
#             return "file not found"
#     file = request.files['testFile']
#     if file:
#         image_stream = BytesIO(file.read())
#         ecg = ECG()
#         ecg_user_image = ecg.getImage(image_stream)
#         ecg_user_gray_image = ecg.GrayImage(ecg_user_image)
#         dividing_leads = ecg.DividingLeads(ecg_user_image)
#         ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
#         ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
#         ecg_1dsignal = ecg.CombineConvert1Dsignal(ec_signal_extraction)
#         ecg_final = ecg.DimensionalReduction(ecg_1dsignal)
#         ecgmodel = ecg.ModelLoad_predict(ecg_final)
#     return ecgmodel

@app.route('/dashboard')
def render_dashboard():

    if not isLogin:
        return render_template('/dashboard.html',patient=get_patient_details())
    else:
        return render_template('/login.html')

if __name__ == '__main__':
    app.run(debug=True)