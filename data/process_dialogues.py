import re
import os

def process_dialogues():
    # Obtenir le chemin absolu
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, 'cornell movie-dialogs corpus', 'movie_lines.txt')
    output_file = os.path.join(current_dir, 'movie_dialogues.txt')
    
    print(f"Lecture du fichier : {input_file}")
    
    # Lire le fichier des dialogues
    with open(input_file, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    
    print(f"Nombre de lignes lues : {len(lines)}")
    
    # Traiter les dialogues
    dialogues = []
    for line in lines:
        # Utiliser le bon séparateur
        parts = line.strip().split(' +++$+++ ')
        if len(parts) >= 5:
            text = parts[4].strip()
            if text:
                dialogues.append(text)
    
    print(f"Nombre de dialogues extraits : {len(dialogues)}")
    
    # Sauvegarder les dialogues
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in dialogues:
            f.write(text + '\n')
    
    print(f"Fichier sauvegardé dans : {output_file}")
    print(f"Nombre total de dialogues : {len(dialogues)}")

if __name__ == "__main__":
    process_dialogues() 