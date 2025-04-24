# Oumllack-GPT

Un chatbot basé sur GPT-2 et entraîné sur des dialogues de films.

## Description

Oumllack-GPT est un chatbot qui utilise un modèle GPT-2 fine-tuné sur des dialogues de films pour générer des réponses naturelles et engageantes. L'application est construite avec Streamlit pour une interface utilisateur intuitive et moderne.

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/Oumllack/Oumllack-GPT.git
cd Oumllack-GPT
```

2. Créez un environnement virtuel et installez les dépendances :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
pip install -r requirements.txt
```

3. Téléchargez le modèle :
Le modèle n'est pas inclus dans ce dépôt en raison de sa taille. Vous pouvez :
- Utiliser le modèle GPT-2 par défaut
- Télécharger notre modèle fine-tuné (lien à venir)

## Utilisation

1. Activez l'environnement virtuel :
```bash
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

2. Lancez l'application :
```bash
streamlit run app.py
```

3. Ouvrez votre navigateur à l'adresse : http://localhost:8501

## Structure du Projet

```
.
├── app.py              # Application Streamlit
├── requirements.txt    # Dépendances Python
└── chatbot_model/     # Dossier du modèle (non inclus dans le dépôt)
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT. 