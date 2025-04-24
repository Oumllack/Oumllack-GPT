import os
import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
import re

# Configuration de la page (DOIT ÊTRE EN PREMIER)
st.set_page_config(
    page_title="Oumllack-GPT",
    page_icon="🤖",
    layout="wide"
)

# Vérification du modèle
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot_model")
if not os.path.exists(MODEL_PATH):
    st.error("❌ Le dossier du modèle n'existe pas!")
    st.stop()

required_files = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt"
]

missing_files = []
for file in required_files:
    if not os.path.exists(os.path.join(MODEL_PATH, file)):
        missing_files.append(file)

if missing_files:
    st.error(f"❌ Fichiers manquants dans le dossier du modèle: {', '.join(missing_files)}")
    st.stop()

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

def clean_and_validate_french(text):
    # Nettoyer les caractères spéciaux et les balises HTML
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Liste de mots français très courants
    french_markers = {
        'articles': ['le', 'la', 'les', 'un', 'une', 'des'],
        'pronouns': ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles'],
        'verbs': ['est', 'sont', 'avoir', 'être', 'faire', 'dire', 'voir', 'vouloir'],
        'prepositions': ['à', 'de', 'dans', 'sur', 'pour', 'avec', 'par'],
        'conjunctions': ['et', 'ou', 'mais', 'donc', 'car', 'si']
    }
    
    words = text.lower().split()
    if len(words) < 3:
        return None
        
    # Compter les marqueurs français par catégorie
    scores = {}
    for category, markers in french_markers.items():
        matches = sum(1 for word in words if word in markers)
        scores[category] = matches / len(words)
    
    # Vérifier si au moins 3 catégories ont des scores positifs
    if sum(1 for score in scores.values() if score > 0) < 3:
        return None
        
    # Vérifier la cohérence de la phrase
    if not any(word in french_markers['articles'] + french_markers['pronouns'] for word in words[:2]):
        return None
        
    return text

@st.cache_resource
def load_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True)
        
        # Configuration spécifique du tokenizer
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        model.resize_token_embeddings(len(tokenizer))
        
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            st.success("✅ Modèle chargé sur GPU")
        else:
            st.info("ℹ️ Modèle chargé sur CPU")
            
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return None, None

def generate_response(prompt, max_attempts=5):
    try:
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            return "❌ Erreur: Le modèle n'a pas pu être chargé."
        
        # Contexte enrichi pour guider le modèle
        contexts = [
            "Tu es un assistant francophone qui répond toujours en français correct. ",
            "En tant qu'assistant français, je vais répondre à votre question. ",
            "Je suis là pour vous aider en français. Voici ma réponse : ",
            "Permettez-moi de vous répondre en français. ",
            "En tant qu'assistant francophone, je vous propose cette réponse : "
        ]
        
        for attempt in range(max_attempts):
            with st.spinner("🤔 Génération de la réponse..."):
                # Alterner entre différents contextes
                context = contexts[attempt % len(contexts)]
                input_text = f"{context}[USER] {prompt.strip()} [ASSISTANT] "
                
                encoded = tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                attention_mask = torch.ones_like(encoded["input_ids"])
                
                if torch.cuda.is_available():
                    encoded["input_ids"] = encoded["input_ids"].cuda()
                    attention_mask = attention_mask.cuda()
                
                outputs = model.generate(
                    encoded["input_ids"],
                    attention_mask=attention_mask,
                    max_length=100,
                    temperature=0.8,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=30,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.5,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.replace(context, "").replace("[USER]", "").replace("[ASSISTANT]", "")
                response = response.replace(prompt, "").strip()
                
                # Valider et nettoyer la réponse
                cleaned_response = clean_and_validate_french(response)
                if cleaned_response:
                    return cleaned_response
        
        return "Je m'excuse, je n'arrive pas à générer une réponse cohérente en français. Pourriez-vous reformuler votre question ?"
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération: {str(e)}")
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

# Interface simple
user_input = st.text_input("💬 Votre message:")
if user_input:
    response = generate_response(user_input)
    st.write("🤖 Réponse:", response) 