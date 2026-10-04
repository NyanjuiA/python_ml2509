"""
================================================================================
Python script to demonstrate calculation of TF-IDF (Term Frequency Inverse Document Frequency)
================================================================================


Requirements
--------------------------
 pip install scikit-learn

Author : Nyanjui
Date   : 02 October 2026
"""
# ================================================================================
# SECTION 0: Import the required module
# ================================================================================
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data from document(s)
documents = [
   "Text mining transforms unstructured text in structured data.",
   "TF-IDF is a technique used in text mining",
   "Text analysis often involves TF-IDF"
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Compute TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (terms) and their corresponding TF-IDF scores
feature_names = vectorizer.fit_transform(documents)
tfidf_scores = tfidf_matrix.toarray()

# Display results
print("Feature names: ", feature_names)
print("TF-IDF scores: ", tfidf_scores)

# 🎉🎉🔥 Congratulations!!! You've successfully installed the DATA SCIENCE Module. Bye. 💯

