import streamlit as st
import pywhatkit
import datetime
import time

st.set_page_config(page_title="WhatsApp AutoSender", layout="centered")

st.title("💬 Auto-Envoi WhatsApp avec pywhatkit")
st.write("Programmez vos messages WhatsApp facilement.")

# Inputs utilisateur
numero = st.text_input("📱 Numéro WhatsApp (avec +)", "+1")
message = st.text_area("✍️ Message à envoyer", "Bonjour, ceci est un message automatique.")
heure = st.number_input("🕐 Heure d'envoi (24h)", min_value=0, max_value=23, value=datetime.datetime.now().hour)
minute = st.number_input("🕐 Minute d'envoi", min_value=0, max_value=59, value=(datetime.datetime.now().minute + 2) % 60)

# Bouton pour envoyer
if st.button("📤 Programmer l’envoi"):
    try:
        st.success(f"Message programmé pour {heure:02d}:{minute:02d} à {numero}")
        pywhatkit.sendwhatmsg(numero, message, int(heure), int(minute))
    except Exception as e:
        st.error(f"Erreur : {e}")
