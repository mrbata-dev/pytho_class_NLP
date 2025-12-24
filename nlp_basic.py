# text = "HEllo WORLD!, how are you?, I am fine, you?, I am fine, he is eating apple, are they?"
# text = text.lower()
# print(text)

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
text = "HEllo WORLD!, how are you?, I am fine, you?, I am fine, he is eating apple, are they? better, playing,  having"
text = text.lower()
print(text)

#Tokenization
tokens = word_tokenize(text)
print("Tokenization--->",tokens)


#Remove Stopwords
stop_words = set(stopwords.words('english'))
tokens=[word for word in tokens if word not in stop_words]
print("Stop words --->",tokens)

#Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatizer_words = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
print("lemmatizer--->", lemmatizer_words)



