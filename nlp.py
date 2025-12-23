# from sqlalchemy import create_engine
# engine = create_engine(
#     "mysql+pymysql://root:admin@localhost:3306/python_class"
# )

# # To insert data into the db
# import pandas as pd


# data = {
#     "name": ["A", "B", "C"],
#     "age": [10, 20, 30],
#     "gender": ["M", "F", "M"],
#     "class": ["A", "B", "C"]
# }


# df_new = pd.DataFrame(data)
# df_new.to_sql("student", con=engine, if_exists="replace", index=False)


# query = "SELECT * FROM student"
# df=pd.read_sql(query, engine)
# print(df)


#Setps:
# 1) Data Collection
# 2) To kenization
#3) stop word removal
# 4) stemming / lemmatization
# 5) Embeddings
#6) Model

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("Kali Linux is awesome!"))


#Nlp
#1.text collection
#2. Text preprocessing/cleaning
# -lowercasing-"Apple", "apple"
#-removing punctuation
#-removing stopword: 'is', 'a', 'the'
#-stemming/lemmatization-"running"-"run"
# -handing numbers, urls, emojis, special characters
#3 Tokenization"
# -split text into tokens(word, subwords, or characters)
#4. Feature extraction/embeddings
#methods:
#    -bag 0f words(Bow)
#       -TF-IDF
#       word embeddings(word2vec, glove)
#       contextual embeddings(BERT, GPT)
#. model training
#6. Evallllluation
# 7. predicition/deployment
