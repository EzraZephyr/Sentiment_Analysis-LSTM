import torch
import torch.nn as nn
import torch.nn.functional as F

class sentimentmodel(nn.Module):
    def __init__(self, word_count, output):
        super(sentimentmodel, self).__init__()
        self.ebd = nn.Embedding(word_count, 256)
        # Define the embedding layer, mapping the vocabulary size to 256-dimensional vectors

        self.lstm = nn.LSTM(256, 256, 1, batch_first=True)
        # Define the LSTM layer, run one cycle, and set batch_size as the first dimension (good practice)

        self.out = nn.Linear(256, output)
        # Define the fully connected layer

    def forward(self, X, hidden):
        ebd = self.ebd(X)
        # First, pass through the embedding layer to convert word indices to word vectors

        ebd = F.dropout(ebd, p=0.5, training=self.training)
        # Use dropout to randomly deactivate some neurons during training to prevent overfitting
        # Only effective during training

        ebd = ebd.squeeze(1)
        # Remove dimensions of size 1 to simplify tensor shape and fit method processing

        ebd, hidden = self.lstm(ebd, hidden)
        # Enter the LSTM layer for computation

        ebd = ebd[:, -1, :]
        # Take the output of the last time step

        out = self.out(ebd)
        # Pass the last time step through the fully connected layer to get the output

        return torch.sigmoid(out), hidden
        # Map the output values to between 0 and 1 for easier 0 or 1 judgment

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, 256).to(next(self.parameters()).device),
                torch.zeros(1, batch_size, 256).to(next(self.parameters()).device))
        # Initialize and return all-zero states for the hidden and cell layers,
        # ensuring they are computed on the same GPU or CPU
