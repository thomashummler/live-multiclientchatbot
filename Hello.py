import streamlit as st
import openai
import os

API_KEY = os.environ["API_KEY"]

openai.api_key = API_KEY

input = st.text_input("Ask a question")

if st.button("Submit"):

    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": input}])

    st.write(chat_completion.choices[0].message.content)