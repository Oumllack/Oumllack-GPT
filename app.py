import os
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

# Configuration de la page (DOIT ÊTRE EN PREMIER)
st.set_page_config(
    page_title="Oumllack-GPT",
    page_icon="🤖",
    layout="wide"
)

# Afficher le répertoire de travail
st.write(f"Répertoire de travail: {os.getcwd()}")

# Style CSS personnalisé
st.markdown("""
    <style>
    /* Style général */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    
    /* Style de la sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Style des messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        margin-left: auto;
        max-width: 80%;
    }
    
    .chat-message.bot {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        margin-right: auto;
        max-width: 80%;
    }
    
    /* Style des inputs et boutons */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("🤖 Oumllack-GPT")
st.markdown("Un chatbot IA entraîné sur des dialogues de films")

# Sidebar pour les paramètres
with st.sidebar:
    st.header("Paramètres")
    temperature = st.slider("Température", 0.1, 1.0, 0.7, 0.1)
    max_length = st.slider("Longueur maximale", 50, 200, 100, 10)
    
    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("""
    Oumllack-GPT est un chatbot IA entraîné sur des dialogues de films.
    Il utilise le modèle GPT-2 fine-tuné pour générer des réponses naturelles.
    """)

@st.cache_resource
def load_model():
    try:
        # Utiliser le chemin absolu
        model_path = "/Users/spacegreen/Desktop/NLP project/chatbot_model"
        st.write(f"Tentative de chargement depuis: {model_path}")
        
        if not os.path.exists(model_path):
            st.error(f"Dossier non trouvé: {model_path}")
            return None, None
            
        # Vérifier les fichiers
        required_files = ["model.safetensors", "config.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                st.write(f"✅ Fichier trouvé: {file}")
            else:
                st.error(f"❌ Fichier manquant: {file}")
                return None, None
        
        st.write("Chargement du tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
        st.write("Tokenizer chargé")
        
        st.write("Chargement du modèle...")
        model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        st.write("Modèle chargé")
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model.eval()
        
        st.success("Modèle et tokenizer chargés avec succès!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def generate_response(prompt):
    try:
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            return "Erreur: Modèle non chargé"
            
        st.write("Génération de la réponse...")
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=100,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()
    except Exception as e:
        st.error(f"Erreur lors de la génération: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return f"Erreur: {str(e)}"

# Initialisation des états de session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Zone de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("Écrivez votre message ici"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Génération de la réponse..."):
            response = generate_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response}) 