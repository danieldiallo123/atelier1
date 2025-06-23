import streamlit as st
import pandas as pd
import pywhatkit
import datetime
import time

# Configuration de l'interface Streamlit
st.set_page_config(page_title="Auto-Envoi WhatsApp", layout="centered")
st.title("📅 Envoi Programmé de Messages WhatsApp")
st.markdown("Importez un fichier `.xlsx` avec les colonnes : `numero`, `message`, `datetime` (ex: 2025-06-22 14:30)")

# --- Fonction pour corriger les numéros ---
def format_numero(numero):
    numero = str(numero).strip().replace(" ", "")
    if not numero.startswith("+"):
        if len(numero) == 10:  # Numéro canadien sans indicatif
            return "+1" + numero
        else:
            return None
    return numero

# --- Upload du fichier Excel ---
uploaded_file = st.file_uploader("📤 Choisir un fichier Excel", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Vérification des colonnes requises
        if {"numero", "message", "datetime"}.issubset(df.columns):
            # Conversion de la colonne datetime
            df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M", errors='coerce')

            # Formatage des numéros
            df["numero"] = df["numero"].apply(format_numero)

            # Suppression des lignes avec datetime ou numéro invalide
            df = df.dropna(subset=["numero", "datetime"])

            st.success("✅ Fichier chargé avec succès. Messages valides à envoyer :")
            st.dataframe(df)

            if st.button("🚀 Lancer l’envoi des messages"):
                with st.spinner("📨 Envoi en cours..."):
                    for _, row in df.iterrows():
                        numero = row["numero"]
                        message = str(row["message"])
                        send_time = row["datetime"]
                        now = datetime.datetime.now()

                        delay = (send_time - now).total_seconds()

                        if delay < -60:
                            st.warning(f"⏩ Message à {numero} ignoré (date déjà passée : {send_time})")
                            continue

                        if delay > 60:
                            st.info(f"🕒 Attente jusqu'à {send_time.strftime('%Y-%m-%d %H:%M')} pour {numero}")
                            time.sleep(delay - 30)

                        try:
                            pywhatkit.sendwhatmsg(
                                numero,
                                message,
                                send_time.hour,
                                send_time.minute,
                                wait_time=25,
                                tab_close=False
                            )
                            st.success(f"✅ Envoyé à {numero} à {send_time.strftime('%H:%M')}")
                        except Exception as e:
                            st.error(f"❌ Échec d'envoi à {numero} : {e}")
        else:
            st.error("❌ Le fichier doit contenir les colonnes : `numero`, `message`, `datetime`")
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
else:
    st.info("📂 Veuillez importer un fichier Excel pour démarrer.")
