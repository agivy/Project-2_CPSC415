import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
import csv
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_and_preprocess(filename):
    """Load and preprocess the text file."""
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    # Tokenize the text
    # tokens = word_tokenize(text.lower())  # lowercase for consistency
    tokens = [token.lower() for token in text.split()]
    return tokens

class LossCallback(CallbackAny2Vec):
    """Callback to track loss during training."""
    def __init__(self):
        self.epoch = 0
        self.losses = []
        
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            self.losses.append(loss)
        else:
            self.losses.append(loss - self.previous_loss)
        self.previous_loss = loss
        if (self.epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f'Epoch {self.epoch + 1}: Loss {self.losses[-1]:.4f}')
        self.epoch += 1

def train_cbow_model(tokens, vector_size=100, window=2, min_count=1, epochs=100):
    """Train a CBOW model using Word2Vec and track losses."""
    # Initialize callback
    loss_callback = LossCallback()
    
    # Initialize and train model
    model = Word2Vec(sentences=[tokens],
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    sg=1,  # 1 for SG
                    workers=4,
                    compute_loss=True,  # Enable loss computation
                    callbacks=[loss_callback])
    
    # Train the model
    model.train([tokens], total_examples=1, epochs=epochs,
                compute_loss=True, callbacks=[loss_callback])
    
    return model, loss_callback.losses

def plot_training_losses(loss_history1, loss_history2, save_path='training_losses_sg.png'):
    """Plot training losses for both models."""
    plt.figure(figsize=(12, 6))
    
    # Plot both loss curves
    epochs = range(1, len(loss_history1) + 1)
    plt.plot(epochs, loss_history1, 'b-', label='Model 1', linewidth=2)
    plt.plot(epochs, loss_history2, 'r-', label='Model 2', linewidth=2)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining loss plot saved to {save_path}")

def save_model_params(model, loss_history, filename):
    """Save model parameters, word vectors, and loss history."""
    params = {
        'vector_size': model.vector_size,
        'window': model.window,
        'min_count': model.min_count,
        'workers': model.workers,
        'sg': model.sg,
        'vocabulary': dict(model.wv.key_to_index),
        'vectors': {word: model.wv[word].tolist() for word in model.wv.key_to_index},
        'loss_history': loss_history
    }
    
    with open(filename, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Model parameters and loss history saved to {filename}")

def get_word_vector(word, model):
    """Get word vector from model if word exists."""
    try:
        return model.wv[word]
    except KeyError:
        return None

def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    if v1 is None or v2 is None:
        return None
    # Reshape vectors for sklearn's cosine_similarity
    v1_reshaped = v1.reshape(1, -1)
    v2_reshaped = v2.reshape(1, -1)
    return cosine_similarity(v1_reshaped, v2_reshaped)[0][0]

def find_disparate_pairs(vocab1, vocab2, model1, model2):
    """Find pairs of words with different similarities in the two encodings."""
    disparate_pairs = []
    words = list(set(vocab1) & set(vocab2))  # Common words
    
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            word1, word2 = words[i], words[j]
            
            # Get vectors from both models
            v1_e1 = get_word_vector(word1, model1)
            v2_e1 = get_word_vector(word2, model1)
            v1_e2 = get_word_vector(word1, model2)
            v2_e2 = get_word_vector(word2, model2)
            
            if all(v is not None for v in [v1_e1, v2_e1, v1_e2, v2_e2]):
                sim_e1 = compute_cosine_similarity(v1_e1, v2_e1)
                sim_e2 = compute_cosine_similarity(v1_e2, v2_e2)
                
                if sim_e1 is not None and sim_e2 is not None:
                    disparity = abs(sim_e1 - sim_e2)
                    if 0 < sim_e1 < 1 and -1 < sim_e2 < 0:
                        disparate_pairs.append((word1, word2, sim_e1, sim_e2, disparity))
    
    return sorted(disparate_pairs, key=lambda x: x[4], reverse=True)

def save_top_disparities(disparate_pairs, filename='top_100_disparities_gensim_sg.csv', n=1000):
    """Save top N most disparate pairs to a CSV file."""
    top_pairs = disparate_pairs[:n]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Word1', 'Word2', 'Similarity_E1', 'Similarity_E2', 'Disparity'])
        for word1, word2, sim_e1, sim_e2, disparity in top_pairs:
            writer.writerow([word1, word2, sim_e1, sim_e2, disparity])
    
    print(f"\nSaved top {n} disparate pairs to {filename}")

def plot_disparities(disparate_pairs, n=100):
    """Plot disparities for top N pairs."""
    top_pairs = disparate_pairs[:n]
    
    # Extract data
    word_pairs = [f"{pair[0]}-{pair[1]}" for pair in top_pairs]
    disparities = [pair[4] for pair in top_pairs]
    
    # Create plot
    plt.figure(figsize=(15, 10))
    plt.bar(range(len(disparities)), disparities, color='skyblue')
    
    # Customize plot
    plt.title('Word Pair Disparities', fontsize=14, pad=20)
    plt.ylabel('Disparity', fontsize=12)
    plt.xlabel('Word Pairs', fontsize=12)
    plt.xticks(range(len(disparities)), word_pairs, rotation=90, ha='right', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('word_pair_disparities_sg.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Download required NLTK data
    nltk.download('punkt')
    
    # Parameters
    vector_size = 100
    window_size = 2
    epochs = 3000
    
    # Process first document
    print("Processing first document...")
    tokens1 = load_and_preprocess('document1_tokenized.txt')
    
    # Process second document
    print("Processing second document...")
    tokens2 = load_and_preprocess('document2_tokenized.txt')
    
    # Train models
    print("\nTraining first model...")
    model1, loss_history1 = train_cbow_model(tokens1, vector_size=vector_size, 
                                           window=window_size, epochs=epochs)
    
    print("\nTraining second model...")
    model2, loss_history2 = train_cbow_model(tokens2, vector_size=vector_size, 
                                           window=window_size, epochs=epochs)
    
    # Plot training losses
    plot_training_losses(loss_history1, loss_history2)
    
    # Save models and losses
    model1.save("sg_model1.model")
    model2.save("sg_model2.model")
    print("\nModels saved in gensim format")
    
    # Save parameters, vectors, and loss history
    save_model_params(model1, loss_history1, "sg_model1_params.json")
    save_model_params(model2, loss_history2, "sg_model2_params.json")
    
    # Save word vectors
    np.save("cbow_model1_vectors.npy", model1.wv.vectors)
    np.save("cbow_model2_vectors.npy", model2.wv.vectors)
    print("Word vectors saved in numpy format")
    
    # Print loss statistics
    print("\nLoss Statistics Model 1:")
    print(f"Initial loss: {loss_history1[0]:.4f}")
    print(f"Final loss: {loss_history1[-1]:.4f}")
    print(f"Average loss: {np.mean(loss_history1):.4f}")
    print(f"Loss reduction: {((loss_history1[0] - loss_history1[-1])/loss_history1[0])*100:.2f}%")
    
    print("\nLoss Statistics Model 2:")
    print(f"Initial loss: {loss_history2[0]:.4f}")
    print(f"Final loss: {loss_history2[-1]:.4f}")
    print(f"Average loss: {np.mean(loss_history2):.4f}")
    print(f"Loss reduction: {((loss_history2[0] - loss_history2[-1])/loss_history2[0])*100:.2f}%")
    
    # Find disparate pairs
    print("\nFinding disparate pairs...")
    disparate_pairs = find_disparate_pairs(set(tokens1), set(tokens2), model1, model2)
    
    # Save results
    save_top_disparities(disparate_pairs)
    plot_disparities(disparate_pairs)

if __name__ == "__main__":
    main()