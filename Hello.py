import streamlit as st
from openai import OpenAI
import os

API_KEY = os.environ["API_KEY"]



input = st.text_input("Ask a question")


if st.button("Submit"):
    client = OpenAI(
    api_key= API_KEY
    )
    chat_completion = client.chat.completions.create(
         model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": input}]
        )

    st.write(chat_completion.choices[0].message.content)