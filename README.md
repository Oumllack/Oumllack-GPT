# Movie Dialogue Chatbot

A conversational AI chatbot trained on movie dialogues from the Cornell Movie-Dialogs Corpus. This project implements a GPT-based model that can engage in natural conversations inspired by movie dialogues.

## Dataset

The model is trained on the [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), which contains:
- 220,579 conversational exchanges between 10,292 pairs of movie characters
- 9,035 characters from 617 movies
- 304,713 utterances

The dataset is processed to create clean dialogue pairs suitable for training a conversational model.

## Project Structure

```
├── app.py                 # Streamlit web application
├── chatbot_model/         # Trained model checkpoints
├── data/                  # Dataset and processing scripts
│   ├── movie_dialogues.txt    # Processed dialogues
│   └── process_dialogues.py   # Script to process raw dialogues
├── src/                   # Source code
│   ├── chatbot.py         # Chatbot implementation
│   ├── prepare_data.py    # Data preparation utilities
│   ├── preprocess.py      # Text preprocessing functions
│   ├── telegram_bot.py    # Telegram bot integration
│   └── train.py           # Training script
└── requirements.txt       # Project dependencies
```

## Model Architecture

The chatbot is built using the GPT-2 architecture with the following specifications:
- Base model: GPT-2
- Vocabulary size: 50,257 tokens
- Maximum sequence length: 128 tokens
- Training batch size: 8
- Learning rate: 5e-5
- Number of epochs: 3

## Training Process

1. **Data Preparation**
   - Extract dialogues from the Cornell Movie-Dialogs Corpus
   - Clean and preprocess the text
   - Create conversation pairs
   - Tokenize the data using GPT-2 tokenizer

2. **Model Training**
   - Initialize GPT-2 model
   - Fine-tune on movie dialogues
   - Save checkpoints every 1000 steps
   - Monitor training metrics using TensorBoard

3. **Evaluation**
   - Evaluate model performance on test set
   - Generate sample conversations
   - Assess response quality and coherence

## Deployment

The chatbot is deployed in two ways:

1. **Web Interface (Streamlit)**
   - Interactive web application
   - Real-time conversation
   - User-friendly interface

2. **Telegram Bot**
   - Accessible through Telegram
   - Supports group chats
   - Maintains conversation context

## Requirements

```
torch>=1.9.0
transformers>=4.30.0
streamlit>=1.22.0
python-telegram-bot>=20.0
numpy>=1.24.3
pandas>=1.5.3
```

## Usage

1. **Web Interface**
```bash
streamlit run app.py
```

2. **Telegram Bot**
   ```bash
   python src/telegram_bot.py
   ```

## Training Details

- **Hardware**: NVIDIA GPU (recommended)
- **Training Time**: ~2 hours on a single GPU
- **Model Size**: ~500MB
- **Checkpoints**: Saved every 1000 steps
- **Evaluation Metrics**: Perplexity, BLEU score

## Future Improvements

- Implement beam search for better response generation
- Add support for multiple languages
- Integrate with more messaging platforms
- Implement conversation memory
- Add sentiment analysis for better context understanding

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Cornell Movie-Dialogs Corpus for the dataset
- Hugging Face Transformers library
- Streamlit for the web interface
- Python-Telegram-Bot library 