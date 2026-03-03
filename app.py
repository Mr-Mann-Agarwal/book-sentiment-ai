import streamlit as st
import pickle

# Load the brain and the dictionary
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("🤖 Sentiment Analyzer AI")
st.write("Day 6: My first deployed AI model!")

user_input = st.text_input("Enter a review to test:")

if user_input:
    # Use the vectorizer to translate text to numbers
    vec = vectorizer.transform([user_input])
    prediction = model.predict(vec)
    
    # Show the result
    if prediction[0] == "POSITIVE":
        st.success(f"Result: {prediction[0]} ✨")
    else:
        st.error(f"Result: {prediction[0]} ❌")