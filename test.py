import streamlit as st

# Globale Variable für die Chat-Historie
chat_history = []

def add_message(message):
    """Fügt eine neue Nachricht zur Chat-Historie hinzu."""
    chat_history.append(message)

# Streamlit-Anwendung
def main():
    st.title("Einfacher Chat")

    # Eingabefeld für Nachrichten
    new_message = st.text_input("Schreibe eine Nachricht:")

    # Button zum Senden der Nachricht
    if st.button("Senden"):
        add_message(new_message)
        st.text_input("Schreibe eine Nachricht:", value="", key=1)  # Eingabefeld zurücksetzen

    # Anzeigen der Chat-Historie
    st.write("Chat-Historie:")
    for message in chat_history:
        st.text(message)

if __name__ == "__main__":
    main()
