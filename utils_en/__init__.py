import nltk

nltk.download('wordnet')
# Download the wordnet package for lemmatization

nltk.download('stopwords')
# Download the stopwords package to filter out useless words

# Note: If after downloading, you still get an error saying these packages cannot be found,
# check C:\Users\YourName\ and open the hidden folder AppData.
# Then, navigate to Roaming\nltk_data\corpora and locate the zip files for these packages.
# Unzip them into the corpora folder and it should work.
