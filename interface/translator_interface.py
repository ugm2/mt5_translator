import streamlit as st
import requests
import json

def get_translation(sentence):
    params = {'sentences': [sentence]}
    response = requests.post(
        "http://localhost:5001/translate",
        json=params
    )
    try:
        result = response.json()['translations']
    except:
        result = response.json()
    return result

def interface():
    st.header("MT5 Translator English-Spanish")

    sentence = st.text_input("Introduce a sentence either in english or in spanish")

    if sentence:
        with st.spinner("Sending request..."):
            translation = get_translation(sentence)
        
        if translation:
            st.write(translation[0])

interface()