# -------------------------------------------------------------------------
# AUTHOR: Roberto Reyes
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 4.5 hrs
# -----------------------------------------------------------*/
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays


import csv
from sklearn.metrics.pairwise import cosine_similarity


# Read documents from CSV
documents = []
with open('cleaned_documents.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  
    for row in reader:
        documents.append((int(row[0]), row[1]))  
        
def tokenize(document):
    return document.split()

def build_vocab(documents):
    vocab = set()
    for doc in documents:
        vocab.update(tokenize(doc[1]))  
    return sorted(vocab)  

def build_doc_term_matrix(documents, vocab):
    matrix = []
    for doc in documents:
        words = set(tokenize(doc[1]))
        matrix.append([1 if word in words else 0 for word in vocab])
    return matrix


vocab = build_vocab(documents)
doc_term_matrix = build_doc_term_matrix(documents, vocab)


max_similarity = 0
most_similar_docs = (0, 0)

for i in range(len(doc_term_matrix)):
    for j in range(i + 1, len(doc_term_matrix)):
        similarity = cosine_similarity([doc_term_matrix[i]], [doc_term_matrix[j]])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_docs = (documents[i][0], documents[j][0])  


print(f"The most similar documents are document {most_similar_docs[0]} and document {most_similar_docs[1]} with cosine similarity = {max_similarity}")

