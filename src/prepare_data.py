import os
import zipfile
import re

def prepare_movie_dialogues():
    # Chemins des fichiers
    zip_path = "../data/cornell_movie_dialogs.zip"
    output_path = "../data/movie_dialogues.txt"
    
    # Créer le dossier processed s'il n'existe pas
    os.makedirs("../data/processed", exist_ok=True)
    
    # Extraire le fichier ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("../data/processed")
    
    # Lire le fichier des dialogues
    dialogues_file = "../data/processed/cornell movie-dialogs corpus/movie_lines.txt"
    
    # Lire les dialogues ligne par ligne
    dialogues = []
    with open(dialogues_file, 'r', encoding='latin-1') as f:
        for line in f:
            # Utiliser une expression régulière pour extraire le texte
            match = re.split(' \\+\\+\\+\\$\\+\\+\\+ ', line.strip())
            if len(match) >= 5:  # S'assurer qu'il y a au moins 5 colonnes
                text = match[4].strip()
                if text:  # S'assurer que le texte n'est pas vide
                    dialogues.append(text)
    
    # Sauvegarder les dialogues dans un fichier texte
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in dialogues:
            # Nettoyer le texte
            text = text.replace('\n', ' ').strip()
            if text:  # S'assurer que le texte n'est pas vide après nettoyage
                f.write(text + '\n')
    
    print(f"Données préparées et sauvegardées dans {output_path}")
    print(f"Nombre total de dialogues : {len(dialogues)}")

if __name__ == "__main__":
    prepare_movie_dialogues() 