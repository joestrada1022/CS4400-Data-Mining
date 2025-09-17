# -------------------------------------------------------------------------
# AUTHOR: Joshua Estrada
# FILENAME: similarity.py
# SPECIFICATION: 
# FOR: CS 4990 (Data Mining) - Assignment #1
# TIME SPENT: 
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy or pandas.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         documents.append (row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection using the white space as your character delimiter.
#--> add your Python code here

# get all words for binary encoding
all_words_from_doc = set()
for doc_num, words in documents:
   split_words = words.split(' ')
   for word in split_words:
      all_words_from_doc.add(word)
vocab = list(sorted(all_words_from_doc)) # order the set
docTermMatrix = []
for doc_num, words in documents:
  split_words = words.split(' ')
  vector = [0] * len(vocab)
  for i, word in enumerate(vocab):
    if word in split_words:
      vector[i] = 1
  docTermMatrix.append([doc_num, vector])


# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here
highest_sim = -1
most_similar = (None, None)
for i in range(len(docTermMatrix)):
   for j in range(i + 1, len(docTermMatrix)):
      doc1_num, vector1 = docTermMatrix[i]
      doc2_num, vector2 = docTermMatrix[j]

      similarity = cosine_similarity([vector1], [vector2])[0][0]

      if similarity > highest_sim:
         highest_sim = similarity
         most_similar = (doc1_num, doc2_num)



# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
print(f"The most similar documents are document {most_similar[0]} and document {most_similar[1]} with cosine similarity = {highest_sim:.4f}")