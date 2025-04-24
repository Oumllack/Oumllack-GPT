import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path, tokenizer, block_size=64):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def main():
    # Configuration
    model_name = "gpt2"
    output_dir = "../models/chatbot"
    train_file = "../data/movie_dialogues.txt"
    
    # Créer le dossier de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialiser le tokenizer et le modèle
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Ajouter le token de padding si nécessaire
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Charger le dataset
    logger.info("Chargement du dataset...")
    train_dataset = load_dataset(train_file, tokenizer)
    
    # Configurer l'entraînement avec des paramètres optimisés
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=100,
        fp16=False,  # Désactivé car non supporté sur CPU
    )
    
    # Initialiser le data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialiser le trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # Lancer l'entraînement
    logger.info("Début de l'entraînement...")
    trainer.train()
    
    # Sauvegarder le modèle
    logger.info("Sauvegarde du modèle...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Entraînement terminé!")

if __name__ == "__main__":
    main() 