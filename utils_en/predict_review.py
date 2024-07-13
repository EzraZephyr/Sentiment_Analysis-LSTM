import pickle
import keras
import torch
from nltk import word_tokenize
from utils_en.data_loader import clean_text
from utils_en.model import sentimentmodel

def load_model():
    with open('./preprocessed_data/word_to_index.pkl', 'rb') as f:
        word_to_index = pickle.load(f)

    model = sentimentmodel(len(word_to_index), 1)
    model.load_state_dict(torch.load('./model/sentiment_model.pt', map_location=torch.device('cpu')))
    model.eval()
    # Set the model's embedding vocabulary size and output, then load the model and set it to evaluation mode

    return model, word_to_index

def predict_input(model, word_to_index, review):
    sentence = []
    # Used to store the sentence converted to indices

    """review = input("Please enter your review (or type 'exit' to quit): ")
    if review.lower() == 'exit':
        break"""
    # If not using a GUI interface for input, the above is the input method

    text = word_tokenize(clean_text(review))
    # Tokenize the sentence after processing it with regular expressions

    for word in text:
        if word in word_to_index:
            sentence.append(word_to_index[word])
            # If the word is in the vocabulary, add it

        else:
            sentence.append(0)
            # Otherwise, set it to 0

    sentence = keras.preprocessing.sequence.pad_sequences([sentence], maxlen=100, padding='post')
    # Trim the sequence of indices to a fixed length of 100, padding with 0 if necessary

    sentence = torch.tensor(sentence, dtype=torch.long).unsqueeze(0)
    # Convert the sequence of indices to a tensor and add an extra dimension to match the model's input format

    hidden = model.init_hidden(1)
    output, _ = model(sentence, hidden)
    output = output.squeeze()
    # Run the model and remove unnecessary dimensions

    answer = (output >= 0.5).float()
    # Convert the answer to a binary result

    if answer:
        print('This is a positive review.')
        return 'This is a positive review.'
    else:
        print('This is a negative review.')
        return 'This is a negative review.'
        # The return statement is for the GUI popup to receive and display the returned string
        # If not using a GUI, these return statements can be removed

"""def predict_review():
    model, word_to_index = load_model()
    predict_input(model, word_to_index)"""
# If not using a GUI, call this function instead
