# Importing essential libraries and modules

# Apply Python 3.13 compatibility fix first
import compatibility_fix

from flask import Flask, redirect, render_template, request, session, g, url_for
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
try:
    disease_model = ResNet9(3, len(disease_classes))
    disease_model.load_state_dict(torch.load(
        disease_model_path, map_location=torch.device('cpu')))
    disease_model.eval()
    print("Disease model loaded successfully.")
except Exception as e:
    print(f"Error loading disease model: {e}")
    disease_model = None


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest_new.pkl'
try:
    crop_recommendation_model = pickle.load(
        open(crop_recommendation_model_path, 'rb'))
    print("Crop recommendation model loaded successfully.")
except:
    print("Error: Could not load crop recommendation model.")
    crop_recommendation_model = None


# =========================================================================================

# Custom functions for calculations

def predict_image(img, model=disease_model):
    
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    hflip_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(img)).convert('RGB')

    # Test-time augmentation: average logits over a few simple augmentations
    logits_sum = None
    for tfm in (base_transform, hflip_transform):
        img_t = tfm(image)
        img_u = torch.unsqueeze(img_t, 0)
        with torch.no_grad():
            logits = model(img_u)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)

    probs = torch.softmax(logits_sum, dim=1)
    conf, pred_idx = torch.max(probs, dim=1)
    class_name = disease_classes[pred_idx.item()]
    return class_name, float(conf.item())


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
app.secret_key = 'change-this-secret-key'

# ----------------------- Simple i18n setup -----------------------
# Supported languages
SUPPORTED_LANGS = ['en', 'hi']

# Minimal translations for core UI
TRANSLATIONS = {
    'en': {
        'Home': 'Home',
        'Crop': 'Crop',
        'Fertilizer': 'Fertilizer',
        'Disease': 'Disease',
        'Made for farmers by farmers': 'Made for farmers by farmers',
        'Smart Farming Solutions': 'Smart Farming Solutions',
        'Get informed decisions about your farming strategy': 'Get informed decisions about your farming strategy',
        'Get Started': 'Get Started',
        'Learn More': 'Learn More',
        "WHAT WE SOLVE": "WHAT WE SOLVE",
        "Here are some questions we'll answer": "Here are some questions we'll answer",
        'What crop to plant?': 'What crop to plant?',
        'Get recommendations based on soil and climate conditions': 'Get recommendations based on soil and climate conditions',
        'What fertilizer to use?': 'What fertilizer to use?',
        'Find the right nutrients for your soil and crops': 'Find the right nutrients for your soil and crops',
        'What disease affects your crop?': 'What disease affects your crop?',
        'Identify plant diseases with our AI-powered system': 'Identify plant diseases with our AI-powered system',
        'How to cure the disease?': 'How to cure the disease?',
        'Get treatment recommendations and preventive measures': 'Get treatment recommendations and preventive measures',
        'OUR SERVICES': 'OUR SERVICES',
        'How FarmNex Can Help You': 'How FarmNex Can Help You',
        'Crop Recommendation': 'Crop Recommendation',
        'Fertilizer Recommendation': 'Fertilizer Recommendation',
        'Disease Detection': 'Disease Detection',
        'ABOUT US': 'ABOUT US',
        'Improving Agriculture, Improving Lives': 'Improving Agriculture, Improving Lives',
        'Data-Driven Decisions': 'Data-Driven Decisions',
        'Use AI and machine learning to optimize your farming strategy': 'Use AI and machine learning to optimize your farming strategy',
        'Sustainable Farming': 'Sustainable Farming',
        'Environmentally friendly practices for long-term success': 'Environmentally friendly practices for long-term success',
        'Language': 'Language',
        'English': 'English',
        'Hindi': 'Hindi',
        # Crop page
        'Find the most suitable crop for your farm based on soil conditions and climate': 'Find the most suitable crop for your farm based on soil conditions and climate',
        'Nitrogen (N)': 'Nitrogen (N)',
        'Phosphorous (P)': 'Phosphorous (P)',
        'Potassium (K)': 'Potassium (K)',
        'pH Level': 'pH Level',
        'Range: 0-14 (7 is neutral)': 'Range: 0-14 (7 is neutral)',
        'Rainfall (mm)': 'Rainfall (mm)',
        'Annual rainfall in millimeters': 'Annual rainfall in millimeters',
        'Temperature (°C)': 'Temperature (°C)',
        'Average temperature in Celsius': 'Average temperature in Celsius',
        'Humidity (%)': 'Humidity (%)',
        'Relative humidity percentage': 'Relative humidity percentage',
        'State': 'State',
        'City': 'City',
        'Get Recommendation': 'Get Recommendation',
        "Our recommendation system uses machine learning to analyze your inputs and suggest the best crop.": "Our recommendation system uses machine learning to analyze your inputs and suggest the best crop.",
        # Fertilizer page
        'Get personalized fertilizer advice based on soil nutrients and crop type': 'Get personalized fertilizer advice based on soil nutrients and crop type',
        'Crop You Want to Grow': 'Crop You Want to Grow',
        'Select a crop': 'Select a crop',
        'Balanced Nutrients': 'Balanced Nutrients',
        "Optimize your soil's NPK balance": "Optimize your soil's NPK balance",
        'Crop-Specific': 'Crop-Specific',
        "Tailored to your crop's needs": "Tailored to your crop's needs",
        'Sustainable': 'Sustainable',
        'Environmentally conscious advice': 'Environmentally conscious advice',
        # Disease page
        'Plant Disease Detection': 'Plant Disease Detection',
        'Upload an image of your plant to identify potential diseases': 'Upload an image of your plant to identify potential diseases',
        'Upload Plant Image': 'Upload Plant Image',
        'Select a clear image of the affected plant part': 'Select a clear image of the affected plant part',
        'Detect Disease': 'Detect Disease',
        'Image Analysis': 'Image Analysis',
        'Advanced AI-powered detection': 'Advanced AI-powered detection',
        'Disease Identification': 'Disease Identification',
        'Accurate diagnosis': 'Accurate diagnosis',
        'Treatment Advice': 'Treatment Advice',
        'Get remedial suggestions': 'Get remedial suggestions',
        # Result pages
        'Recommendation Results': 'Recommendation Results',
        'You should grow': 'You should grow',
        'in your farm for optimal yield': 'in your farm for optimal yield',
        'This recommendation is based on your soil composition and local climate conditions.': 'This recommendation is based on your soil composition and local climate conditions.',
        'Try Another Prediction': 'Try Another Prediction',
        'Disease Detection Results': 'Disease Detection Results',
        'Diagnosis': 'Diagnosis',
        'Next Steps': 'Next Steps',
        'Early detection is crucial for effective treatment. Consider consulting with a local agricultural expert for specific treatment options.': 'Early detection is crucial for effective treatment. Consider consulting with a local agricultural expert for specific treatment options.',
        'Analyze Another Plant': 'Analyze Another Plant',
        'Why This Matters': 'Why This Matters',
        'Proper fertilization ensures your crops receive the nutrients they need for optimal growth and yield.': 'Proper fertilization ensures your crops receive the nutrients they need for optimal growth and yield.',
        'Try Another Analysis': 'Try Another Analysis'
    },
    'hi': {
        'Home': 'होम',
        'Crop': 'फसल',
        'Fertilizer': 'उर्वरक',
        'Disease': 'रोग',
        'Made for farmers by farmers': 'किसानों द्वारा किसानों के लिए',
        'Smart Farming Solutions': 'स्मार्ट खेती समाधान',
        'Get informed decisions about your farming strategy': 'अपनी खेती की रणनीति के बारे में सूचित निर्णय लें',
        'Get Started': 'शुरू करें',
        'Learn More': 'और जानें',
        "WHAT WE SOLVE": "हम क्या हल करते हैं",
        "Here are some questions we'll answer": "यहाँ कुछ प्रश्न हैं जिनका हम उत्तर देंगे",
        'What crop to plant?': 'कौन सी फसल बोयें?',
        'Get recommendations based on soil and climate conditions': 'मिट्टी और जलवायु के आधार पर सिफारिशें प्राप्त करें',
        'What fertilizer to use?': 'कौन सा उर्वरक उपयोग करें?',
        'Find the right nutrients for your soil and crops': 'अपनी मिट्टी और फसल के लिए सही पोषक तत्व खोजें',
        'What disease affects your crop?': 'आपकी फसल को कौन सा रोग प्रभावित कर रहा है?',
        'Identify plant diseases with our AI-powered system': 'हमारी एआई-संचालित प्रणाली से पौधों के रोग पहचानें',
        'How to cure the disease?': 'रोग का इलाज कैसे करें?',
        'Get treatment recommendations and preventive measures': 'उपचार सुझाव और रोकथाम उपाय प्राप्त करें',
        'OUR SERVICES': 'हमारी सेवाएं',
        'How FarmNex Can Help You': 'FarmNex आपकी कैसे मदद कर सकता है',
        'Crop Recommendation': 'फसल सिफारिश',
        'Fertilizer Recommendation': 'उर्वरक सिफारिश',
        'Disease Detection': 'रोग पहचान',
        'ABOUT US': 'हमारे बारे में',
        'Improving Agriculture, Improving Lives': 'कृषि सुधारें, जीवन सुधारें',
        'Data-Driven Decisions': 'डेटा-आधारित निर्णय',
        'Use AI and machine learning to optimize your farming strategy': 'एआई और मशीन लर्निंग से अपनी खेती की रणनीति अनुकूलित करें',
        'Sustainable Farming': 'सतत खेती',
        'Environmentally friendly practices for long-term success': 'दीर्घकालिक सफलता हेतु पर्यावरण के अनुकूल तरीके',
        'Language': 'भाषा',
        'English': 'अंग्रेज़ी',
        'Hindi': 'हिंदी',
        # Crop page
        'Find the most suitable crop for your farm based on soil conditions and climate': 'मिट्टी और जलवायु के आधार पर अपने खेत के लिए सबसे उपयुक्त फसल खोजें',
        'Nitrogen (N)': 'नाइट्रोजन (N)',
        'Phosphorous (P)': 'फॉस्फोरस (P)',
        'Potassium (K)': 'पोटेशियम (K)',
        'pH Level': 'pH स्तर',
        'Range: 0-14 (7 is neutral)': 'सीमा: 0-14 (7 तटस्थ है)',
        'Rainfall (mm)': 'वर्षा (मिमी)',
        'Annual rainfall in millimeters': 'वार्षिक वर्षा (मिलीमीटर में)',
        'Temperature (°C)': 'तापमान (°C)',
        'Average temperature in Celsius': 'औसत तापमान (सेल्सियस में)',
        'Humidity (%)': 'आर्द्रता (%)',
        'Relative humidity percentage': 'सापेक्ष आर्द्रता प्रतिशत',
        'State': 'राज्य',
        'City': 'शहर',
        'Get Recommendation': 'सिफारिश प्राप्त करें',
        "Our recommendation system uses machine learning to analyze your inputs and suggest the best crop.": "हमारी सिफारिश प्रणाली आपके इनपुट का विश्लेषण करने और सर्वोत्तम फसल सुझाने के लिए मशीन लर्निंग का उपयोग करती है।",
        # Fertilizer page
        'Get personalized fertilizer advice based on soil nutrients and crop type': 'मिट्टी के पोषक तत्वों और फसल के प्रकार के आधार पर उर्वरक सलाह प्राप्त करें',
        'Crop You Want to Grow': 'वह फसल जिसे आप उगाना चाहते हैं',
        'Select a crop': 'फसल चुनें',
        'Balanced Nutrients': 'संतुलित पोषक तत्व',
        "Optimize your soil's NPK balance": "अपनी मिट्टी के NPK संतुलन का अनुकूलन करें",
        'Crop-Specific': 'फसल-विशिष्ट',
        "Tailored to your crop's needs": "आपकी फसल की ज़रूरतों के अनुसार",
        'Sustainable': 'सतत',
        'Environmentally conscious advice': 'पर्यावरण के प्रति जागरूक सलाह',
        # Disease page
        'Plant Disease Detection': 'पौध रोग पहचान',
        'Upload an image of your plant to identify potential diseases': 'संभावित रोगों की पहचान के लिए अपने पौधे की एक छवि अपलोड करें',
        'Upload Plant Image': 'पौधे की छवि अपलोड करें',
        'Select a clear image of the affected plant part': 'प्रभावित पौधे के भाग की स्पष्ट छवि चुनें',
        'Detect Disease': 'रोग पहचानें',
        'Image Analysis': 'छवि विश्लेषण',
        'Advanced AI-powered detection': 'उन्नत एआई-संचालित पहचान',
        'Disease Identification': 'रोग पहचान',
        'Accurate diagnosis': 'सटीक निदान',
        'Treatment Advice': 'उपचार सलाह',
        'Get remedial suggestions': 'उपचारात्मक सुझाव प्राप्त करें',
        # Result pages
        'Recommendation Results': 'सिफारिश परिणाम',
        'You should grow': 'आपको उगानी चाहिए',
        'in your farm for optimal yield': 'सर्वोत्तम पैदावार के लिए अपने खेत में',
        'This recommendation is based on your soil composition and local climate conditions.': 'यह सिफारिश आपकी मिट्टी की संरचना और स्थानीय जलवायु स्थितियों पर आधारित है।',
        'Try Another Prediction': 'एक और पूर्वानुमान आज़माएँ',
        'Disease Detection Results': 'रोग पहचान परिणाम',
        'Diagnosis': 'निदान',
        'Next Steps': 'अगले कदम',
        'Early detection is crucial for effective treatment. Consider consulting with a local agricultural expert for specific treatment options.': 'प्रभावी उपचार के लिए प्रारंभिक पहचान महत्वपूर्ण है। विशिष्ट उपचार विकल्पों के लिए किसी स्थानीय कृषि विशेषज्ञ से परामर्श करें।',
        'Analyze Another Plant': 'एक और पौधे का विश्लेषण करें',
        'Why This Matters': 'यह क्यों महत्वपूर्ण है',
        'Proper fertilization ensures your crops receive the nutrients they need for optimal growth and yield.': 'उचित खाद से आपकी फसलों को सर्वोत्तम वृद्धि और पैदावार के लिए आवश्यक पोषक तत्व मिलते हैं।',
        'Try Another Analysis': 'एक और विश्लेषण आज़माएँ'
    }
}

def translate(text):
    lang = getattr(g, 'lang', None) or session.get('lang', 'en')
    return TRANSLATIONS.get(lang, {}).get(text, text)

@app.before_request
def _set_lang():
    lang = session.get('lang')
    if lang not in SUPPORTED_LANGS:
        session['lang'] = 'en'
    g.lang = session['lang']

@app.route('/set-lang')
def set_lang():
    lang = request.args.get('lang', 'en')
    if lang not in SUPPORTED_LANGS:
        lang = 'en'
    session['lang'] = lang
    next_url = request.args.get('next') or request.referrer or url_for('home')
    return redirect(next_url)

# Make translator available in Jinja
app.jinja_env.globals.update(_=translate)

# home page


@ app.route('/')
def home():
    title = 'FarmNex - Home'
    return render_template('index.html', title=title)

# crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'FarmNex - Crop Recommendation'
    return render_template('crop.html', title=title)

# fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'FarmNex - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)


# ===============================================================================================

# RENDER PREDICTION PAGES
#  crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'FarmNex - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        T = int(request.form['temperature'])
        H = int(request.form['humidity'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")
        
        if crop_recommendation_model is None:
            return render_template('try_again.html', title=title)
        
        # Use ML model for prediction
        data = np.array([[N, P, K, T, H, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        
        print(f"ML Model prediction: {final_prediction}")
        return render_template('crop-result.html', prediction=final_prediction, title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'FarmNex - Fertilizer Suggestion'
    

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)



# disease prediction result page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'FarmNex - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            
            # render disease prediction input page
            
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            pred_class, confidence = predict_image(img)
            prediction = Markup(str(disease_dic.get(pred_class, pred_class)))
            return render_template('disease-result.html', prediction=prediction, confidence=confidence, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False, port=5001)