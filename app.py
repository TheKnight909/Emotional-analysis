import json
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import html
import os
from google.cloud import translate_v2 as translate
import google.auth
from google.oauth2 import service_account
import joblib 
from preProcess import ArabicTextPreprocessor


# Retrieve the API keys from environment variables
openai_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
gcp_credentials = os.getenv("GCP_CREDENTIALS")

# Initialize the Google Translate API client with credentials
def get_credentials():
    if gcp_credentials:
        creds = service_account.Credentials.from_service_account_info(json.loads(gcp_credentials))
    else:
        creds, _ = google.auth.default()
    return creds

# def get_credentials():
#     creds_json = st.secrets["gcp"]["credentials"]
#     if creds_json:
#         creds = service_account.Credentials.from_service_account_info(json.loads(creds_json))
#     else:
#         creds, _ = google.auth.default()
#     return creds

# Initialize the Google Translate API client with credentials
translator = translate.Client(credentials=get_credentials())


# Initialize the Google Translate API client
# translator = translate.Client()

# Use st.cache_resource to cache the model loading
@st.cache_resource
def get_model():
    # Load tokenizer and model for emotion detection
    model = AutoModelForSequenceClassification.from_pretrained("TheKnight115/Finetuned_MarBERT_Arabic_Emotional_Analysis")
    new_tokenizer = AutoTokenizer.from_pretrained("TheKnight115/fine-tuned-bert-base-uncased")
    new_model = AutoModelForSequenceClassification.from_pretrained("TheKnight115/fine-tuned-bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("TheKnight115/Finetuned_MarBERT_Arabic_Emotional_Analysis")
    preprocessor = ArabicTextPreprocessor()
    SVM = joblib.load('SVM_model.pkl')
    KNN = joblib.load('KNN_model.pkl')
    RF = joblib.load('RF_model.pkl')

    model.eval()  # Set the model to evaluation mode
    new_model.eval()
    return model, new_model, new_tokenizer, tokenizer, preprocessor, SVM, KNN, RF

model, new_model, new_tokenizer, tokenizer, preprocessor, SVM, KNN, RF = get_model()  # Load model when script runs

def classify_traditional_approach_emotion(text):
    processed_text = preprocessor.preprocess_text(text)
    return processed_text

def SVM_Classified_Emotion(text):
    return SVM.predict([text])[0]

def KNN_Classified_Emotion(text):
    return KNN.predict([text])[0]

def RF_Classified_Emotion(text):
    return RF.predict([text])[0]



def analyze_text_emotion(text, target_language="en"):
    """Translate the text to English and analyze the emotion of the translated text."""
    translation = translator.translate(text, target_language)
    translated_text = html.unescape(translation['translatedText'])
    inputs = new_tokenizer(translated_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = new_model(**inputs)
    
    probabilities = torch.softmax(outputs.logits, dim=-1)
    labels = new_model.config.id2label
    
    # Get the top emotion
    top_emotion_index = torch.argmax(probabilities, dim=-1).item()
    top_emotion = labels[top_emotion_index]
    top_probability = probabilities[0, top_emotion_index].item() * 100
    
    return f"{top_emotion} ({top_probability:.2f}%)\nThe Translated text: {translated_text}"

# Retrieve the API key securely
# openai_key = st.secrets["api_keys"]["openai_key"]
# google_api_key = st.secrets["api_keys"]["google_api_key"]

# Explicitly set the API key for OpenAI (not recommended for production)
client = OpenAI(api_key=openai_key)

# Gemini API key (also not recommended for production storage)
API_KEY = google_api_key



def classify_emotion_bert(text):
    # Generate inputs from the text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    max_prob, predicted_class = torch.max(probabilities, dim=1)
    max_prob_percentage = max_prob.item() * 100  # Convert to percentage
    labels = ['Joy', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise']  # Adjust labels as per your model
    predicted_emotion = labels[predicted_class.item()]
    return f"{predicted_emotion} ({max_prob_percentage:.2f}%)"


def classify_emotion_openai(text):
    try:
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify the following text into Joy, Sadness, Anger, Fear, Disgust, or Surprise with percentage so the output will only be the emotion with it percentage"},
                {"role": "user", "content": text}
            ],
            stream=True
        )
        result = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content
        return result
    except Exception as e:
        return f"An error occurred: {e}"

def classify_emotion_gemini(text):
    genai.configure(api_key=API_KEY)
    generation_config = {
    "temperature": 0,  # Controls randomness (0 for deterministic, 1 for randomness)
    "top_p": 1,      # Probability distribution over tokens (1 for maximum likelihood)
    "top_k": 1       # Restricts generation to top k most likely tokens
}
    safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    # Add other safety settings as required
]
    model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings, generation_config=generation_config)
    chat = model.start_chat(history=[])
    
    response = response = chat.send_message("Simply classify the following text snippet into only one of six emotions: joy, sadness, anger, fear, disgust, or surprise with the with percentage so the output will only be the emotion with it percentage and don't say to me i don't know" + text)
    return response.text

def main():
    st.title('Emotion Classifier')

    user_input = st.text_area("Enter your text here:", height=150)
    if st.button('Classify Emotion'):
        if user_input:
            with st.spinner('Analyzing...'):
                openai_result = classify_emotion_openai(user_input)
                gemini_result = classify_emotion_gemini(user_input)
                marbert_result = classify_emotion_bert(user_input)
                translated_emotion_result = analyze_text_emotion(user_input)
                Pre_processed_text = classify_traditional_approach_emotion(user_input)
                svm_result = SVM_Classified_Emotion(Pre_processed_text)
                knn_result = KNN_Classified_Emotion(Pre_processed_text)
                rf_result = RF_Classified_Emotion(Pre_processed_text)
            # Creating two columns for layout
            col1, col2 = st.columns([2, 2])
            
            with col1:
                st.write(f"**GPT Classified Emotion:** {openai_result}")
                st.write(f"**Gemini Classified Emotion:** {gemini_result}")
                st.write(f"**Our Finetuned MarBERT Classified Emotion:** {marbert_result}")
                st.write(f"**English bert Classified Emotion:** {translated_emotion_result}")
            
            with col2:
                st.write(f"**SVM Classified Emotion:** {svm_result}")
                st.write(f"**Random Forest Classified Emotion:** {rf_result}")
                st.write(f"**KNN Classified Emotion:** {knn_result}")
        else:
            st.error("Please enter some text to analyze.")

if __name__ == '__main__':
    main()
