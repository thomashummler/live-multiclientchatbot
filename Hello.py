import streamlit as st
import openai

# Assuming you have set up st.secrets correctly
openai.api_key = st.secrets["openai_api_key"]

input = st.text_input("Ask a question")

if st.button("Submit"):
    chat_completion = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": input}]
    )

    st.write(chat_completion.choices[0].message.content)
