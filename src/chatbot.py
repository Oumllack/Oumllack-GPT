import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Modèle chargé sur {self.device}")

    def generate_response(self, input_text, max_length=100):
        # Encoder l'entrée
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Générer la réponse
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Décoder la réponse
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main():
    chatbot = Chatbot()
    print("Chatbot initialisé. Tapez 'quit' pour quitter.")
    
    while True:
        user_input = input("Vous: ")
        if user_input.lower() == 'quit':
            break
            
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main() 