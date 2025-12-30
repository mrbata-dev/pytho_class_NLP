from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

#Step 1 :Load the txt doc
def load_document(path):
    "---Load text from file----"
    with open(path, 'r', encoding='utf-8') as f:
        # print("file read====", f.read())
        return f.read()
    
# text = load_document('/home/secret/Documents/pythonVirtualEnv/file.txt')

#Step 2: Parse Q&A Pairs
def parse_qa_pairs(text):
    """
    Docstring for parse_qa_pairs
    
    :param text: Description
    """
    qa_pairs = []
    pattern = r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\nQ:|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    for q, a in matches:
        qa_pairs.append((q.strip(), a.strip()))
    return qa_pairs

# print(parse_qa_pairs(text))
# parse_qa_pairs(text)


def build_index(questions, model):
    embeddings = model.encode(questions, convert_to_numpy = True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, embeddings

# search best answer
def get_answer(user_question, model, index, qa_pairs):
    query_embedding = model.encode([user_question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k =1)
    best_match = indices[0][0]
    return qa_pairs[best_match][1]

# Main Chatbot
def chatbot():
    text = load_document("/home/secret/Documents/pythonVirtualEnv/file.txt")
    qa_PAIRS = parse_qa_pairs(text)
    questions = [q for q, a in qa_PAIRS]

    #load embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    #build FAISS index
    index, _ = build_index(questions, embed_model)
    print("Chatbot is ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        answer = get_answer(user_input, embed_model, index, qa_PAIRS)
        print("Bot: ", answer)

#Run
if __name__ == "__main__":
    chatbot()