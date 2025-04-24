import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from chatbot import Chatbot

# Configuration du logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialisation du chatbot
chatbot = Chatbot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /start"""
    await update.message.reply_text(
        "Bonjour! Je suis votre chatbot NLP. Posez-moi une question!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /help"""
    await update.message.reply_text(
        "Je suis un chatbot entraîné sur des dialogues de films. "
        "Posez-moi une question et je ferai de mon mieux pour y répondre!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gestion des messages"""
    try:
        response = chatbot.generate_response(update.message.text)
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Erreur lors de la génération de la réponse: {e}")
        await update.message.reply_text(
            "Désolé, j'ai rencontré une erreur. Veuillez réessayer."
        )

def main():
    """Fonction principale"""
    # Remplacez 'YOUR_BOT_TOKEN' par votre token Telegram
    application = Application.builder().token('YOUR_BOT_TOKEN').build()

    # Ajout des handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Démarrage du bot
    application.run_polling()

if __name__ == '__main__':
    main() 