import streamlit as st
import openai
import os
# Assuming you have set up st.secrets correctly
openai.api_key = os.environ["API_KEY"]

input = st.text_input("Ask a question")

if st.button("Submit"):
    chat_completion = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": input}]
    )

    st.write(chat_completion.choices[0].message.content)
