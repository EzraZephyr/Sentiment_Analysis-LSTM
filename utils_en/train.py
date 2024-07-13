import pickle
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils_en.data_process import data_process
from utils_en.model import sentimentmodel

def train():
    with open('./preprocessed_data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('./preprocessed_data/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    with open('./preprocessed_data/word_count.pkl', 'rb') as f:
        word_count = pickle.load(f)

    train_process = data_process(X_train, y_train)
    # Create a data processing object

    model = sentimentmodel(word_count, 1)
    # Initialize the model, setting output to 1

    dataloader = DataLoader(train_process, batch_size=32, shuffle=True, drop_last=True)
    # Create a data loader, with batches of 32 and drop the last batch if it's less than 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Run on GPU if available, otherwise on CPU

    criterion = nn.BCELoss()
    # Use binary cross-entropy loss for better measuring the error between predicted probabilities and target values

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Use Adam optimizer with adaptive learning rate to update parameters

    epoch = 35
    # Set the number of training epochs

    train_log = './model/training.log'
    file = open(train_log, 'w')
    # Save training logs (good practice)

    for epoch in range(epoch):
        epoch_idx = 0
        total_loss = 0.0
        start_time = time.time()
        # epoch_idx keeps track of the number of batches processed for averaging loss
        # total_loss calculates the total loss for each training epoch
        # Record the time to calculate the duration of each training epoch

        for X, y in dataloader:

            X, y = X.to(device), y.to(device)
            # Move data and target values to the available device

            hidden = model.init_hidden(batch_size=X.size(0))
            # Initialize hidden states

            output, hidden = model(X, hidden)
            # Forward pass to get the updated output and hidden states

            output = output.squeeze()
            # Remove unnecessary dimensions

            optimizer.zero_grad()
            # Clear gradients to prevent accumulation

            loss = criterion(output, y)
            # Calculate the loss

            total_loss += loss.item()
            # Accumulate the loss, note to use item() to extract the scalar value, otherwise it's a tensor

            loss.backward()
            # Backpropagation

            optimizer.step()
            # Update model parameters

            epoch_idx += 1

        message = 'Epoch:{}, Loss:{:.4f}, Time{:.2f}'.format(epoch + 1, total_loss / epoch_idx, time.time() - start_time)
        file.write(message + '\n')
        print(message)
        # Print and output the training information for each epoch to identify any issues during training

    file.close()
    torch.save(model.state_dict(), './model/sentiment_model.pt')
    # Close the log file and save the trained model
