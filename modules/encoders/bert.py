import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForMaskedLM, BertModel
from torch.optim import AdamW
import numpy as np

def build_vocab_from_traces(full_train_traces):
    """
    Build vocabulary from training traces.
    
    Parameters:
    - full_train_traces: List of activity sequences (training data)
    
    Returns:
    - vocab: Dictionary mapping activities to token IDs
    - inv_vocab: Inverse vocabulary (token IDs to activities)
    """
    all_activities = sorted(list(set([act for seq in full_train_traces for act in seq])))
    
    vocab = {act: i + 5 for i, act in enumerate(all_activities)}
    vocab['[PAD]'] = 0
    vocab['[MASK]'] = 1
    vocab['[CLS]'] = 2
    vocab['[SEP]'] = 3
    vocab['[UNK]'] = 4
    
    inv_vocab = {v: k for k, v in vocab.items()}
    
    return vocab, inv_vocab


def tokenize(seq, vocab, max_len):
    """
    Tokenize a single sequence.
    
    Parameters:
    - seq: List of activities
    - vocab: Vocabulary dictionary
    - max_len: Maximum sequence length
    
    Returns:
    - input_ids: Tensor of token IDs
    - attention_mask: Tensor of attention mask (1 for real tokens, 0 for padding)
    """
    seq_tokens = [vocab['[CLS]']] + [vocab.get(act, vocab['[UNK]']) for act in seq[-(max_len-1):]]
    
    pad_len = max_len - len(seq_tokens)
    input_ids = seq_tokens + [vocab['[PAD]']] * pad_len
    
    attention_mask = [1] * len(seq_tokens) + [0] * pad_len
    
    return torch.tensor(input_ids), torch.tensor(attention_mask)

class LogDataset(Dataset):
    """PyTorch Dataset for activity sequences."""
    
    def __init__(self, traces, vocab, max_len=150):
        self.traces = traces
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return tokenize(self.traces[idx], self.vocab, self.max_len)

def create_bert_config(vocab_size, hidden_size=128, num_layers=4, num_heads=4):
    """
    Create BERT configuration.
    
    Parameters:
    - vocab_size: Size of vocabulary
    - hidden_size: Hidden layer dimensionality
    - num_layers: Number of transformer layers
    - num_heads: Number of attention heads
    
    Returns:
    - BertConfig object
    """
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4
    )
    return config


def pretrain_bert(full_traces, vocab, epochs=5, batch_size=32, max_len=150,
                  hidden_size=128, num_layers=4, num_heads=4, lr=1e-4):
    """
    Pre-train BERT model using masked language modeling.
    
    Parameters:
    - full_traces: List of activity sequences for pre-training
    - vocab: Vocabulary dictionary
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    - max_len: Maximum sequence length
    - hidden_size: BERT hidden dimension
    - num_layers: Number of BERT layers
    - num_heads: Number of attention heads
    - lr: Learning rate
    
    Returns:
    - Trained BERT encoder (BertModel, not BertForMaskedLM)
    """
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    config = create_bert_config(len(vocab), hidden_size, num_layers, num_heads)
    model = BertForMaskedLM(config).to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    dataset = LogDataset(full_traces, vocab, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Starting BERT Pre-training...")
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = input_ids.clone()
            
            # Mask 15% of tokens (excluding special tokens)
            rand = torch.rand(input_ids.shape, device=device)
            mask_arr = (rand < 0.15) & (input_ids > 4)
            input_ids[mask_arr] = vocab['[MASK]']
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    print("Pre-training complete.")
    return model.bert  # Return encoder only

def bert_embed_data(bert_encoder, df, vocab, max_len=150, batch_size=32):
    """
    Extract BERT embeddings from activity sequences.
    
    Parameters:
    - bert_encoder: Pre-trained BERT encoder (BertModel)
    - df: DataFrame with 'subtrace' column
    - vocab: Vocabulary dictionary
    - max_len: Maximum sequence length
    - batch_size: Batch size for inference (for speed)
    
    Returns:
    - X: Embedding matrix (n_samples, hidden_size)
    - y: Target labels (next_activity values)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_encoder = bert_encoder.to(device)

    bert_encoder.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            batch_input_ids = []
            batch_attention_mask = []
            
            for _, row in batch_df.iterrows():
                input_ids, attention_mask = tokenize(row['subtrace'], vocab, max_len)
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
            
            # Stack into batches
            batch_input_ids = torch.stack(batch_input_ids).to(device)
            batch_attention_mask = torch.stack(batch_attention_mask).to(device)
            
            outputs = bert_encoder(batch_input_ids, attention_mask=batch_attention_mask)
            
            # Extract embedding from last real token position for each sequence
            for j in range(len(batch_df)):
                last_real_idx = batch_attention_mask[j].sum() - 1
                emb = outputs.last_hidden_state[j, last_real_idx, :].cpu().numpy()
                embeddings.append(emb)
    
    X = np.array(embeddings)
    y = df['next_activity'].values
    
    return X, y



def prepare_bert_features(bert_encoder, train_df, test_df, vocab, max_len=150, batch_size=32):
    """
    Complete pipeline: embed train and test data.
    
    Parameters:
    - bert_encoder: Pre-trained BERT encoder
    - train_df: Training DataFrame
    - test_df: Test DataFrame
    - vocab: Vocabulary dictionary
    - max_len: Maximum sequence length
    - batch_size: Batch size for inference
    
    Returns:
    - X_train, X_test: Feature matrices
    - y_train, y_test: Target labels
    """
    X_train, y_train = bert_embed_data(bert_encoder, train_df, vocab, max_len, batch_size)
    X_test, y_test = bert_embed_data(bert_encoder, test_df, vocab, max_len, batch_size)
    
    return X_train, X_test, y_train, y_test