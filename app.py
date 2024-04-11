import streamlit as st
import pickle
import spacy
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import emoji


st.set_page_config(
    page_title="Emotions",
    page_icon="👋",
)





lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
def preprocess_sentence(sen):
    doc = nlp(sen)
    new_sen = [token.lemma_ for token in doc ]
    return " ".join(new_sen)

st.title("Emotion Classification")


with open("emotions.pkl", "rb") as f:
    rf_classifier, tfidf_vectorizer = pickle.load(f)

text_input = st.text_input("Enter text:", "")
classify_button = st.button("Classify Emotion")
ref_sen = preprocess_sentence(text_input)


emo = {0 : f'Sad {emoji.emojize(":disappointed_face:")}', 1: f'Joy {emoji.emojize(":smiling_face_with_smiling_eyes:")}', 2: f'Fear {emoji.emojize(":fearful_face:")}', 3: f'anger {emoji.emojize(":angry_face:")}' , 4 : f'love {emoji.emojize(":red_heart:")}', 5 : f'surprise {emoji.emojize(":surprised_face:")}'}





if classify_button:
    if text_input:
        
        X_input = tfidf_vectorizer.transform([ref_sen])
        
        emotion = rf_classifier.predict(X_input)[0]
        st.write(f"Emotion: {emo[emotion]}")
    else:
        st.warning("Please enter text to classify emotion.")
