import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import csv
import matplotlib.pyplot as plt

def load_and_preprocess(filename):
    """Load and preprocess the text file."""
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    tokens = [token.lower() for token in text.split()]
    return tokens

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBOWModel, self).__init__()
        
        # Initialize embeddings with Xavier/Glorot initialization
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        nn.init.xavier_uniform_(self.embeddings.weight)
        
        # Add batch normalization
        self.batch_norm1 = nn.BatchNorm1d(embed_size)
        
        # Add a deeper architecture
        self.hidden1 = nn.Linear(embed_size, embed_size * 2)
        self.hidden2 = nn.Linear(embed_size * 2, embed_size)
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.xavier_uniform_(self.hidden2.weight)
        
        # Add more batch normalization
        self.batch_norm2 = nn.BatchNorm1d(embed_size * 2)
        self.batch_norm3 = nn.BatchNorm1d(embed_size)
        
        # Output layer
        self.output = nn.Linear(embed_size, vocab_size)
        nn.init.xavier_uniform_(self.output.weight)
        
        # Add dropout with lower rate
        self.dropout = nn.Dropout(0.1)
        
        # Add ReLU activation
        self.relu = nn.ReLU()
        
    def forward(self, context):
        if len(context.shape) == 1:
            context = context.unsqueeze(0)
            
        # Get embeddings and average context words
        context_embeds = self.embeddings(context)
        context_embeds = torch.mean(context_embeds, dim=1)
        
        # First batch norm
        x = self.batch_norm1(context_embeds)
        
        # First hidden layer
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.batch_norm3(x)
        x = self.dropout(x)
        
        # Output layer
        output = self.output(x)
        return output

def create_training_data(tokens, context_size, word_to_index):
    """Create training data pairs of (context, target) from tokens."""
    data = []
    for i in range(context_size, len(tokens) - context_size):
        context_words = tokens[i-context_size:i] + tokens[i+1:i+context_size+1]
        context = torch.tensor([word_to_index[word] for word in context_words])
        target = torch.tensor([word_to_index[tokens[i]]])
        data.append((context, target))
    return data

def train_cbow(data, vocab_size, embed_size, learning_rate=0.001, epochs=100):
    """Train the CBOW model."""
    model = CBOWModel(vocab_size, embed_size)
    criterion = nn.CrossEntropyLoss()
    
    # Use Adam optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Use cosine annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                    T_max=epochs,
                                                    eta_min=1e-6)
    
    # Create mini-batches
    batch_size = 32
    num_batches = max(1, len(data) // batch_size)  # Ensure at least 1 batch
    loss_history = []
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        
        # Shuffle data at start of epoch
        np.random.shuffle(data)
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if not batch:  # Skip empty batches
                continue
                
            contexts = torch.stack([item[0] for item in batch])
            targets = torch.cat([item[1] for item in batch])
            
            optimizer.zero_grad()
            outputs = model(contexts)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
    
    return model, loss_history

def plot_training_losses(loss_history1, loss_history2, save_path='training_losses.png'):
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
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTraining loss plot saved to {save_path}")

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    if torch.is_tensor(v1):
        v1 = v1.detach().numpy()
    if torch.is_tensor(v2):
        v2 = v2.detach().numpy()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_word_embedding(word, model, word_to_index):
    """Get the embedding vector for a word."""
    if word in word_to_index:
        index = torch.tensor([word_to_index[word]])
        with torch.no_grad():  # No need to track gradients here
            model.eval()  # Set model to evaluation mode
            return model.embeddings(index).numpy()[0]
    return None

def find_disparate_pairs(vocab, model1, model2, word_to_index1, word_to_index2):
    """Find pairs of words with different similarities in the two encodings."""
    disparities = []
    common_words = set(word_to_index1.keys()) & set(word_to_index2.keys())
    words_list = list(common_words)
    
    for i in range(len(words_list)):
        for j in range(i + 1, len(words_list)):
            word1, word2 = words_list[i], words_list[j]
            
            v1_e1 = get_word_embedding(word1, model1, word_to_index1)
            v2_e1 = get_word_embedding(word2, model1, word_to_index1)
            v1_e2 = get_word_embedding(word1, model2, word_to_index2)
            v2_e2 = get_word_embedding(word2, model2, word_to_index2)
            
            if all(v is not None for v in [v1_e1, v2_e1, v1_e2, v2_e2]):
                sim_e1 = cosine_similarity(v1_e1, v2_e1)
                sim_e2 = cosine_similarity(v1_e2, v2_e2)
                disparity = abs(sim_e1 - sim_e2)
                
                if 0 < sim_e1 < 1 and -1 < sim_e2 < 0:
                    disparities.append((word1, word2, sim_e1, sim_e2, disparity))
    
    return sorted(disparities, key=lambda x: x[4], reverse=True)

def save_top_disparities(disparate_pairs, filename='top_100_disparities.csv', n=100):
    """
    Save top N most disparate pairs to a CSV file.
    """
    # Get top N pairs
    top_pairs = disparate_pairs[:n]
    
    # Write to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Word1', 'Word2', 'Similarity_E1', 'Similarity_E2', 'Disparity'])
        # Write data
        for word1, word2, sim_e1, sim_e2, disparity in top_pairs:
            writer.writerow([word1, word2, sim_e1, sim_e2, disparity])
    
    print(f"\nSaved top {n} disparate pairs to {filename}")

def main():
    # Parameters
    context_size = 2
    embed_size = 100  # As specified
    
    print("Processing first document...")
    tokens1 = load_and_preprocess('document1_tokenized.txt')
    vocab1 = set(tokens1)
    word_to_index1 = {word: i for i, word in enumerate(vocab1)}
    data1 = create_training_data(tokens1, context_size, word_to_index1)
    
    print("Processing second document...")
    tokens2 = load_and_preprocess('document2_tokenized.txt')
    vocab2 = set(tokens2)
    word_to_index2 = {word: i for i, word in enumerate(vocab2)}
    data2 = create_training_data(tokens2, context_size, word_to_index2)
    
    print(f"Vocabulary sizes: Doc1 = {len(vocab1)}, Doc2 = {len(vocab2)}")
    print(f"Training examples: Doc1 = {len(data1)}, Doc2 = {len(data2)}")
    
    print("\nTraining first model...")
    model1, loss_history1 = train_cbow(data1, len(vocab1), embed_size)
    torch.save(model1.state_dict(), 'cbow_model1_state.pt')
    
    print("\nTraining second model...")
    model2, loss_history2 = train_cbow(data2, len(vocab2), embed_size)
    torch.save(model2.state_dict(), 'cbow_model2_state.pt')
    
    # Plot training losses
    plot_training_losses(loss_history1, loss_history2)
    
    print("\nFinding disparate pairs...")
    disparate_pairs = find_disparate_pairs(
        vocab1 & vocab2, model1, model2, word_to_index1, word_to_index2
    )
    
    save_top_disparities(disparate_pairs)
    
    print("\nTop 100 most disparate pairs:")
    for i, (word1, word2, sim_e1, sim_e2, disparity) in enumerate(disparate_pairs[:100]):
        print(f"\nPair {i+1}:")
        print(f"Words: '{word1}' and '{word2}'")
        print(f"Similarity in Encoding 1: {sim_e1:.4f}")
        print(f"Similarity in Encoding 2: {sim_e2:.4f}")
        print(f"Disparity: {disparity:.4f}")
        
def main_infer():
    # Parameters
    context_size = 2
    embed_size = 100
    
    print("Processing first document...")
    tokens1 = load_and_preprocess('document1_tokenized.txt')
    vocab1 = set(tokens1)
    word_to_index1 = {word: i for i, word in enumerate(vocab1)}
    
    print("Processing second document...")
    tokens2 = load_and_preprocess('document2_tokenized.txt')
    vocab2 = set(tokens2)
    word_to_index2 = {word: i for i, word in enumerate(vocab2)}
    
    print(f"Vocabulary sizes: Doc1 = {len(vocab1)}, Doc2 = {len(vocab2)}")
    
    # Initialize models with correct vocabulary sizes
    model1 = CBOWModel(len(vocab1), embed_size)
    model2 = CBOWModel(len(vocab2), embed_size)
    
    # Load saved state dictionaries
    print("\nLoading first model...")
    model1.load_state_dict(torch.load('cbow_model1_state.pt'))
    model1.eval()  # Set to evaluation mode
    
    print("\nLoading second model...")
    model2.load_state_dict(torch.load('cbow_model2_state.pt'))
    model2.eval()  # Set to evaluation mode
    
    print("\nFinding disparate pairs...")
    disparate_pairs = find_disparate_pairs(
        vocab1 & vocab2, model1, model2, word_to_index1, word_to_index2
    )
    
    save_top_disparities(disparate_pairs)
    
    print("\nTop 100 most disparate pairs:")
    for i, (word1, word2, sim_e1, sim_e2, disparity) in enumerate(disparate_pairs[:100]):
        print(f"\nPair {i+1}:")
        print(f"Words: '{word1}' and '{word2}'")
        print(f"Similarity in Encoding 1: {sim_e1:.4f}")
        print(f"Similarity in Encoding 2: {sim_e2:.4f}")
        print(f"Disparity: {disparity:.4f}")

if __name__ == "__main__":
    main_infer()