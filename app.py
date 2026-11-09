from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

word_index=imdb.get_word_index()
reverse_word_index={value : key for key,value in word_index.items()}


model=load_model("rnn_imdb.h5")

def preprocess_text(text):
    words=text.lower().split()
    encoder_review=[word_index.get(word,2) + 3 for word in words]
    pad_review=pad_sequences([encoder_review],maxlen=500)
    return pad_review

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment="positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment,prediction[0][0]

st.title("IMDB sent analysis")
st.write("enter you review ")

user_input=st.text_area('Movie revierw')

if st.button('Classify'):
    sentiment,score=predict_sentiment(user_input)
    st.write(sentiment)
    st.write(score)
else:
    st.write("pls write the review")