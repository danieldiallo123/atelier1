import streamlit as st
import pandas as pd
import pywhatkit
import time

st.set_page_config(page_title="WhatsApp AutoSender", layout="wide")
st.title("📲 Auto-envoi de messages WhatsApp via Excel")

# Téléversement du fichier Excel
uploaded_file = st.file_uploader("📤 Importer un fichier Excel (.xlsx) contenant les messages à envoyer", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # Vérification des colonnes
        required_cols = {"numero", "message", "heure", "minute"}
        if not required_cols.issubset(df.columns):
            st.error(f"Le fichier doit contenir les colonnes : {', '.join(required_cols)}")
        else:
            st.success("✅ Fichier chargé avec succès ! Voici les messages à envoyer :")
            st.dataframe(df)

            if st.button("🚀 Lancer l’envoi des messages"):
                with st.spinner("Envoi en cours..."):
                    for index, row in df.iterrows():
                        try:
                            numero = str(row["numero"])
                            message = str(row["message"])
                            heure = int(row["heure"])
                            minute = int(row["minute"])

                            # Envoi avec délai contrôlé (évite le spam)
                            pywhatkit.sendwhatmsg(numero, message, heure, minute, wait_time=10, tab_close=True)
                            st.success(f"✅ Message à {numero} prévu pour {heure:02d}:{minute:02d}")

                        except Exception as e:
                            st.error(f"❌ Erreur avec {row['numero']}: {e}")
                            continue
    except Exception as e:
        st.error(f"Erreur de lecture du fichier Excel : {e}")
else:
    st.info("Veuillez importer un fichier Excel avec les colonnes : numero, message, heure, minute.")
