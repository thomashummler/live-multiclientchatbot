# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger

def main():
    st.title("Einfache Chat-Anwendung")

    # Chathistorie initialisieren, wenn nicht vorhanden
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Chat-Nachrichteneingabefeld
    with st.form("chat_form"):
        user_input = st.text_input("Schreiben Sie Ihre Nachricht:")
        submit_button = st.form_submit_button("Senden")

    # Nachricht zur Chathistorie hinzufügen
    if submit_button and user_input:
        st.session_state['chat_history'].append(f"Sie: {user_input}")

    # Chathistorie anzeigen
    for message in st.session_state['chat_history']:
        st.text(message)

if __name__ == "__main__":
    main()

