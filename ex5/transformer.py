import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from dataset import DataHandler



class NewGELU(torch.nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))




class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size                
        #### YOUR CODE HERE ####
        # TIP: 
        # It is common practive to initialze a single Linear layer to map each token to its query, key, and value, i.e. nn.Linear(self.n_embd, 3 * self.n_embd)
        # After applying the linear layer on a token embedding you can split the layer's output to key, query, and value
        # The output key/query/value is of dimension n_embd, in practice this includes the embeddings for all heads, 
        # therefore, embedding = [embd_1, embd_2, .. embd_nheads]. You can rearange as you please in the forward pass.

        # Dimension de chaque tête
        self.d_head = n_embd // n_head

        # Linear layer to compute query, key, and value in a single pass
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)  # Output: [q, k, v]

        # Linear layer for the output projection
        self.out_proj = nn.Linear(n_embd, n_embd)

        self.register_buffer(
            "mask",
            torch.full((block_size, block_size), float("-inf")).triu(1)
        )

    def forward(self, x):
        #### YOUR CODE HERE ####
        # Compute queries, keys, and values. Expected shape [batch_size, n_heads, sequence_length n_embd/n_head]
        
        # Compute normalized attention matrix (Q@K.T)/sqrt(d_k), Expected shape [batch_size, n_heads, sequence_length, sequence_length]
        # NOTE: the dimension d_k refers to the embedding dimension of the keys which is n_embd/num_heads 

        # Mask, this is casual self-attention, you need to mask the score of each token with the tokens that come after it in the sequence
        # Fill all values above the diagonal with -float('inf'), this ensures these entries will be zeroed after softmax

        # Apply softmax on each row of the masked normalized attention matrix and perform matrix multiplication with the values
        # Expected shape [batch_size, n_heads, sequence_length, n_embd/n_head]

        # Re-Assemble all head outputs side by side. Expected shape [batch_side, sequence_length, n_embd]
        
        # output projection
        B, T, C = x.size()

        qkv = self.qkv_proj(x)  # Shape: [B, T, 3 * C]
        q, k, v = torch.split(qkv, C, dim=-1)

        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B, nh, T, dh]
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B, nh, T, dh]
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B, nh, T, dh]

        # Attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, nh, T, T]

        # Apply mask
        attn_scores = attn_scores + self.mask[:T, :T]

        # Softmax normalization
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Attention output
        attn_output = attn_probs @ v  # [B, nh, T, dh]

        # Recombine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]

        # Output projection
        y = self.out_proj(attn_output)  # [B, T, C]
        return y



class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """


    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, self.n_embd),
            wpe = nn.Embedding(block_size, self.n_embd),            
            h = nn.ModuleList([Block(n_head, n_embd, block_size) for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)



    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits





def train_model(
        train_path,
        test_path=None,
        model=None,                        
        block_size=10,
        n_layer=3,
        n_head=3,
        n_embd=48,
        learning_rate=3e-4,
        batch_size=64,
        epochs=10
):            
                    
    
    data_handler = DataHandler(train_path, test_path, block_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = data_handler.get_vocab_size()
    if model is None:
        model = GPT(n_layer, n_head, n_embd, vocab_size, block_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    print('Using device:', device)


    trainset = data_handler.get_dataset('train')
    testset = data_handler.get_dataset('test')
    
    # setup the dataloader
    train_loader = DataLoader(
        trainset,
        sampler=torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=int(1e5)),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,        
    )     
    if testset:       
        test_loader = DataLoader(
            testset,
            sampler=torch.utils.data.RandomSampler(testset, replacement=False, num_samples=int(1e4)),
            shuffle=False,
            pin_memory=True,
            batch_size=batch_size,            
        )
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for ep in range(epochs):
        model.train()
        total_loss = 0
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, batch in enumerate(tqdm(train_loader)):            
            #### YOUR CODE HERE ####
            inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Propagation avant
            optimizer.zero_grad()  # On réinitialise les gradients
            logits = model(inputs)

            loss = criterion(logits.view(-1, vocab_size),
                             labels.view(-1))
            total_loss += loss.item()

            # Propagation arrière et optimisation
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = logits.max(dim=-1)
            total_train += labels.size(0) * labels.size(1)
            correct_train += (predicted == labels).sum().item()

            # Calculer la perte et l'accuracy pour le train set
        train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {ep + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")
            

        if testset:
            model.eval()  # Passage en mode évaluation
            total_test_loss = 0
            running_test_loss = 0.0
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for i, batch in enumerate(tqdm(test_loader)):
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    logits = model(inputs)
                    loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
                    total_test_loss += loss.item()

                    running_test_loss += loss.item()
                    _, predicted = logits.max(dim=-1)
                    total_test += labels.size(0) * labels.size(1)
                    correct_test += (predicted == labels).sum().item()

            test_loss = running_test_loss / len(test_loader)
            test_accuracy = correct_test / total_test
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            print(f"Test Loss: {total_test_loss / len(test_loader)}")

            # Complete the sentence:
        model.eval()
        with torch.no_grad():

            for i in range(3):
                sentence = "the "
                for i in range(30):
                    tokens = torch.tensor(data_handler.encoder(sentence[-block_size:]))[None,:].to(device)
                    logits = model(tokens)
                    predictions = logits[:, -1, :]  # Prédictions pour le dernier token

                    # Appliquer la température
                    probabilities = torch.softmax(predictions, dim=-1)

                    # Échantillonnage parmi les probabilités
                    next_token = torch.multinomial(probabilities, 1).item()

                    # Convertir le token en mot
                    next_word = data_handler.decoder([next_token])
                    sentence += next_word
                print('new_sentence: ', sentence)

            print("Generate with Top-k-sampling")
            # Comple the sentence only considering the top k characters when sampling:
            for i in range(3):
                sentence = "the "
                for i in range(30):
                    tokens = torch.tensor(data_handler.encoder(sentence[-block_size:]))[None,:].to(device)
                    logits = model(tokens)
                    predictions = logits[:, -1, :]  # Prédictions pour le dernier token

                    # Obtenir les k meilleures prédictions
                    k = 5  # Choisir le nombre de meilleurs tokens (ajuste ce nombre si nécessaire)
                    top_k_probs, top_k_indices = torch.topk(torch.softmax(predictions, dim=-1), k)

                    # Échantillonnage parmi les k meilleurs tokens
                    next_token = torch.multinomial(top_k_probs, 1).item()
                    next_token = top_k_indices[0, next_token].item()  # Choisir le token parmi les k meilleurs

                    # Convertir le token en mot
                    next_word = data_handler.decoder([next_token])
                    sentence += next_word
                print('new_sentence: ', sentence)

    plt.figure(figsize=(12, 6))

    # Plot de la perte
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot de l'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(epochs), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()




if __name__=="__main__":
    torch.manual_seed(42)
    train_model('train_shakespeare.txt','test_shakespeare.txt')
    

