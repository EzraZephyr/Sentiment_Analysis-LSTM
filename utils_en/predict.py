import pickle
import torch
from torch.utils.data import DataLoader
from utils_en.data_process import data_process
from utils_en.model import sentimentmodel

def predict():
    word_count, X_test, y_test = load()

    test_process = data_process(X_test, y_test)
    test_loader = DataLoader(test_process, batch_size=32, shuffle=False)
    model = sentimentmodel(word_count, 1)
    model.load_state_dict(torch.load('./model/sentiment_model.pt', map_location=torch.device('cpu')))
    # Load the trained model

    model.eval()
    # Set the model to evaluation mode

    total_correct = 0
    total_count = 0

    with torch.no_grad():
        # Disable gradient calculation, as it's not necessary during testing

        for X, y in test_loader:
            hidden = model.init_hidden(batch_size=X.size(0))
            # Initialize hidden states

            outputs, _ = model(X, hidden)
            # Forward pass, hidden state is not needed as we don't backpropagate

            outputs = outputs.squeeze()
            # Remove unnecessary dimensions

            y_pred = (outputs >= 0.5).float()
            # Convert output probabilities to binary results (0 or 1)

            correct = (y_pred == y).sum().item()
            # Calculate the number of correct predictions

            total_correct += correct
            total_count += y.size(0)

    accuracy = total_correct / total_count
    print(f'Accuracy: {accuracy*100:.2f}%')
    # Calculate and print the accuracy

def load():
    with open('./preprocessed_data/word_count.pkl', 'rb') as f:
        word_count = pickle.load(f)
    with open('./preprocessed_data/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('./preprocessed_data/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    return word_count, X_test, y_test
