import pandas as pd
import os

def load_movie_lines():
    """Charge les lignes de dialogue des films"""
    lines = {}
    with open('data/cornell movie-dialogs corpus/movie_lines.txt', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                line_id = parts[0]
                text = parts[4].strip()
                lines[line_id] = text
    return lines

def load_conversations():
    """Charge les conversations"""
    conversations = []
    with open('data/cornell movie-dialogs corpus/movie_conversations.txt', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                line_ids = eval(parts[3])
                conversations.append(line_ids)
    return conversations

def create_training_data():
    """Crée les données d'entraînement"""
    lines = load_movie_lines()
    conversations = load_conversations()
    
    # Créer des paires de dialogues
    training_data = []
    for conv in conversations:
        for i in range(len(conv) - 1):
            input_text = lines[conv[i]]
            target_text = lines[conv[i + 1]]
            training_data.append(f"{input_text}\t{target_text}")
    
    # Sauvegarder les données
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/movie_dialogues.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(training_data))

if __name__ == "__main__":
    create_training_data()
    print("Prétraitement terminé. Les données sont sauvegardées dans data/processed/movie_dialogues.txt") 