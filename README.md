# Bigram-Transformer-Model

A state-of-the-art Bigram Language Model that leverages the power of PyTorch and transformer mechanisms to understand and generate coherent text sequences. This project stands out for its efficient tokenization, embedding techniques, and the pioneering use of MPS device optimization.

## Dependencies
- Python 3.8+
- PyTorch 1.9.0
- torchvision 0.10.0
- torchtext 0.10.0
- numpy 1.21.0

_(Note: Version numbers mentioned are the ones tested on. Later versions might also work.)_

## Setup & Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Bigram-Transformer-Model.git
    cd bigram-transformer-model
    ```

2. **Install dependencies:**
    Using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Steps to Create the Transformer Model:
1. **Data Preparation:**
    - Source a text dataset (`wizard_of_oz_plaintext.txt` used in this project).
    - Tokenize the dataset to convert text into integer tokens.

2. **Model Architecture Design:**
    - Initialize a Bigram Language Model class inheriting `nn.Module` from PyTorch.
    - Define the token embedding table using `nn.Embedding`.

3. **Model Forward Pass:**
    - Pass the tokenized input through the embedding table to get logits.
    - Compute the cross-entropy loss if targets are provided.

4. **Text Generation:**
    - Feed in a context (array of token indices).
    - For each new token to be generated, pass the current sequence to the model to get logits.
    - Apply softmax to logits, sample a next token, and append to the sequence.

5. **Model Training:**
    - Split data into training and validation sets.
    - Define an optimizer (`AdamW` used in this project).
    - Iterate over data batches, compute the loss, backpropagate, and update model weights.

6. **Evaluation and Deployment:**
    - Estimate loss on both training and validation datasets.
    - Generate coherent text sequences using the trained model.

## Usage:
After setup, run the main script to train the model and generate text:
```bash
python main.py
```

## Contribution:
Feel free to raise issues or submit PRs for any enhancements, bug-fixes, or feature additions.


---

_(Note: This README is a template for the given project. Ensure to adapt links, usernames, and other specifics before deploying to GitHub.)_
