import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# Load the Keras model
model = keras.models.load_model('my_model.h5')
vocab_size=500
oov_tok='<OOV>'
max_len=50
token=Tokenizer(num_words=vocab_size,oov_token=oov_tok)
word_index=token.word_index
word_index
# Function to preprocess text
def preprocess_text(text):
    # Your preprocessing steps here (tokenization, padding, etc.)
    return text

# Function to predict if text is spam or not
def predict_spam(predict_msg, padding_type='post'):
    data = pd.read_csv('dataset.csv')
    ham_msg = data[data.text_type =='ham']
    spam_msg = data[data.text_type=='spam']
    ham_msg=ham_msg.sample(n=len(spam_msg),random_state=42)
    balanced_data = pd.concat([ham_msg, spam_msg]).reset_index(drop=True)
    balanced_data.head()
    balanced_data['label']=balanced_data['text_type'].map({'ham':0,'spam':1})
    train_msg, test_msg, train_labels, test_labels =train_test_split(balanced_data['text'],balanced_data['label'],test_size=0.2,random_state=434)
    token.fit_on_texts(train_msg)

    new_seq = token.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =50,
                      padding = padding_type,
                      truncating='post')
    return model.predict(padded)
    
    

# Streamlit UI
def main():
    st.title("Spam Classification App")
    user_input = st.text_input("Enter text to check for spam:")
    
    if st.button("Check for Spam"):
        if user_input.strip() == "":
            st.error("Please enter some text.")
        else:
            l = []
            l.append(user_input)
            prediction = predict_spam(l)
            if prediction[0] > 0.5:
                st.error("This is spam! ")
            else:
                st.success("This is not spam.")
            st.write("Score :", prediction[0])

if __name__ == "__main__":
    main()
