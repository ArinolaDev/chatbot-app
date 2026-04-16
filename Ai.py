import streamlit as st
import numpy as np
import json
import random
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')



# --- 1. Load Model and Intents ---
model = load_model('chat_model.h5')
with open('intent.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

words = sorted(set([lemmatizer.lemmatize(w.lower())
                    for intent in intents['intents']
                    for pattern in intent['patterns']
                    for w in tokenizer.tokenize(pattern) if w.isalnum()]))
classes = sorted([intent['tag'] for intent in intents['intents']])

# --- 2. Product Catalog (Fixed Syntax) ---
products = [
    {
        "name": "iPhone 14",
        "price": "₦950,000",
        "image": "https://r.listwr.me/Sm4ZIN",
        "link": "https://r.listwr.me/FtKW3j"
    },
    {
        "name": "Samsung Galaxy S23",
        "price": "₦850,000",
        "image": "https://r.listwr.me/G3yrwE",
        "link": "https://r.listwr.me/N85ihi"
    },
    {
        "name": "HP Laptop",
        "price": "₦1005278",
        "image": "https://r.listwr.me/pMtaGm",
        "link": "https://r.listwr.me/m8hawW"
    }
]

if "initialized" not in st.session_state:
    st.session_state.show_products = False
    st.session_state.history = [("bot", "Hello! Welcome to Arinola E-Commerce Store. How can I help you today?")]
    st.session_state.initialized = True

# --- 4. Helper Functions ---
def bag_of_words(text):
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens if w.isalnum()]
    return np.array([1 if w in tokens else 0 for w in words])

def predict_class(text):
    bow = bag_of_words(text)
    res = model.predict(np.array([bow]), verbose=0)[0]
    return classes[np.argmax(res)], res[np.argmax(res)]

def get_response(user_input):
    intent, confidence = predict_class(user_input)
    if confidence > 0.5:
        for i in intents['intents']:
            if i['tag'] == intent:
                if intent == "products":
                    return "Sure! Check out our latest products below."
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that. 😅"

def submit():
    user_input = st.session_state.input_text
    if user_input:
        response = get_response(user_input)
        intent, confidence = predict_class(user_input)


        st.session_state.history.append(("user", user_input))
        
        if intent == "products" and confidence > 0.5:
            st.session_state.history.append(("bot", "Here are our products ✨"))
            st.session_state.show_products = True
        else:
            st.session_state.history.append(("bot", response))
            st.session_state.show_products = False

        st.session_state.input_text = ""

# --- 5. Streamlit Layout & CSS ---
st.set_page_config(page_title="Arinola E-Commerce Chatbot", page_icon="🛒")

st.markdown("""
<style>
    .chat-header {
        background-color: #f68b1e;
        color: white;
        padding: 15px;
        border-radius: 10px 10px 0 0;
        text-align: center;
        font-weight: bold;
    }
    .user-msg {
        background-color: #f68b1e !important;
        color: white !important;
        padding: 10px 15px;
        border-radius: 15px 15px 0px 15px;
        margin: 5px 0px 5px auto;
        width: fit-content;
        max-width: 80% !important;
        text-align: right;
    }
    .bot-msg {
        background-color: #f1f1f1;
        color: black;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0px;
        margin: 5px 0px;
        width: fit-content;
        max-width: 80%;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# --- 6. The UI Construction ---
st.markdown('<div class="chat-header">ARINOLA SUPPORT CHATBOT</div>', unsafe_allow_html=True)

with st.container(border=True):
    # Scrollable chat area
    chat_box = st.container(height=400, border=False)
    with chat_box:
        for sender, msg in st.session_state.history:
            if sender == "user":
                st.markdown(f'<div class="user-msg">{msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-msg">{msg}</div>', unsafe_allow_html=True)



    if st.session_state.show_products and len(st.session_state.history) > 1:
        st.markdown("---")
        p_cols = st.columns(len(products))
        for idx, item in enumerate(products):
            with p_cols[idx]:
                st.image(item["image"], use_container_width=True)
                st.caption(f"**{item['name']}**\n{item['price']}")
                st.link_button("Buy Now", item["link"], use_container_width=True)

    st.text_input("Send your message here...", key="input_text", on_change=submit)


    

