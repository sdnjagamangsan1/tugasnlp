import streamlit as st
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

nlp = spacy.load("en_core_web_sm")

text_data = [
    "I love the new iPhone 13. The camera is amazing!",
    "The customer service at this store is terrible.",
    "I don't have any opinion on this laptop brand."
]
sentiment_labels = ["positive", "negative", "neutral"]

named_entities_data = []
for text in text_data:
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    named_entities_data.append(' '.join(named_entities))

vectorizer = CountVectorizer()
vectorizer.fit(named_entities_data)
X = vectorizer.transform(named_entities_data)

lr = LogisticRegression()
lr.fit(X, sentiment_labels)

st.title('Aplikasi Praktikum NLP Ida Hafizah')
st.write('Selamat datang di aplikasi praktikum berbasis Streamlit!')

name = st.text_input('Masukkan nama Anda:')
if name:
    st.write(f'Halo, {name}!')

user_input = st.text_area("Masukkan kalimat untuk analisis sentimen:")

if st.button("Analisis"):
    if user_input:
        doc = nlp(user_input)
        named_entities = [ent.text for ent in doc.ents]
        user_named_entities_data = ' '.join(named_entities)
        
        user_X = vectorizer.transform([user_named_entities_data])
        
        prediction = lr.predict(user_X)
        
        st.write(f"Prediksi Sentimen: {prediction[0]}")
    else:
        st.write("Silakan masukkan kalimat untuk analisis sentimen.")

st.write("Berikut Contoh Prediksi Sentimen yang dapat dicoba:")
new_text_data = [
    "I love the new iPhone 13. The camera is amazing!",
    "The customer service at this store is terrible.",
    "I don't have any opinion on this laptop brand."
]

new_named_entities_data = []
for text in new_text_data:
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    new_named_entities_data.append(' '.join(named_entities))

new_X = vectorizer.transform(new_named_entities_data)
predictions = lr.predict(new_X)

for text, pred in zip(new_text_data, predictions):
    st.write(f"Teks: {text}")
