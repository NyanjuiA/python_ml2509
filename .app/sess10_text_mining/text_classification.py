"""
================================================================================
Python script to demonstrate text classification for a restaurant's food/meal reviews
================================================================================


Requirements
--------------------------
 pip install matplotlib nltk numpy scikit-learn

Author : Nyanjui
Date   : 02 October 2026
"""
# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import matplotlib.pyplot as plt
import nltk
import numpy as np
import re
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB

# ================================================================================
# SECTION 1: Function to check and download the required NTLK data if not present
# ================================================================================
def download_if_missing(resource_name: str, resource_path: str) -> None:
   try:
      nltk.data.find(resource_path)
   except LookupError:
      nltk.download(resource_name)

# Download the 'stopwords' and 'wordnet' corpora
download_if_missing('stopwords','corpora/stopwords')
download_if_missing('wordnet','corpora/wordnet')

# ================================================================================
# SECTION 2: Sample data (food/meal reviews from customers and their labels)
# ================================================================================

texts = [
   "The pasta was absolutely delicious! I'd order it again.",
   "Terrible meal. The chicken was dry and tasteless.",
   "Amazing flavors and perfect presentation. A delightful experience.",
   "The burger was undercooked and greasy. Very disappointing.",
   "Loved the sushi platter! Fresh and flavorful.",
   "Awful service and bland food. Will not return.",
   "Best steak I've had in years. Cooked to perfection!",
   "The soup was too salty and lacked depth.",
   "Incredible dessert selection. The chocolate mousse was divine!",
   "Nothing special. The meal felt overpriced and mediocre.",
   "I enjoyed every bite of the lasagna. Rich and hearty.",
   "The fish was overcooked and rubbery.",
   "Fresh ingredients and vibrant taste. Highly recommend the salad.",
   "The fries were cold and soggy. Not impressed.",
   "That curry had just the right amount of spice. Loved it!",
   "One of the worst dining experiences I've had. Avoid this place.",
   "The pancakes were fluffy and packed with flavor.",
   "The sandwich was soggy and falling apart.",
   "Elegant plating and bold flavors. A gourmet treat.",
   "The rice was undercooked and crunchy. Disappointing.",
   "Perfect brunch spot! Tasty food and cozy atmosphere.",
   "The noodles were bland and overcooked.",
   "Loved the variety and quality of the buffet.",
   "Disgusting. Found a hair in my food.",
   "Fantastic pizza with a crispy crust and fresh toppings.",
   "The pasta was absolutely delicious! I loved every bite.",
   "Great flavors and perfect seasoning, a must-try dish!",
   "An excellent pizza with fresh, high-quality ingredients.",
   "The sushi was incredible—so fresh and well-prepared.",
   "Best burger I’ve ever had! Juicy and full of flavor.",
   "The dessert was heavenly, especially the chocolate cake.",
   "Amazing service and the steak was cooked to perfection.",
   "The tacos were flavorful and had the perfect crunch.",
   "A fantastic breakfast with fluffy pancakes and crispy bacon.",
   "The curry was rich and aromatic, just like homemade.",
   "The salad was fresh, vibrant, and dressed perfectly.",
   "This ramen bowl is divine—broth is rich and noodles are firm.",
   "The seafood platter was fresh and tasted like the ocean.",
   "The lasagna was cheesy, saucy, and absolutely satisfying.",
   "The croissants were buttery, flaky, and melt-in-your-mouth.",
   "Horrible meal. The chicken was dry and flavorless.",
   "Terrible service and the soup was cold.",
   "Overpriced and tasteless—would not recommend.",
   "The fish was overcooked and smelled bad.",
   "Disgusting. The rice was undercooked and hard.",
   "The coffee was burnt and undrinkable.",
   "The fries were soggy and the burger was dry.",
   "Awful experience. The pasta was mushy and bland.",
   "The sandwich was stale and had barely any filling.",
   "The dessert was way too sweet and artificial-tasting.",
]

labels = [
   "positive", "negative", "positive", "negative", "positive",
   "negative", "positive", "negative", "positive", "negative",
   "positive", "negative", "positive", "negative", "positive",
   "negative", "positive", "negative", "positive", "negative",
   "positive", "negative", "positive", "negative", "positive",
   "positive", "positive", "positive", "positive", "positive",
   "positive", "positive", "positive", "positive", "positive",
   "positive", "positive", "positive", "positive", "positive",
   "negative", "negative", "negative", "negative", "negative",
   "negative", "negative", "negative", "negative", "negative",
]

# ================================================================================
# SECTION 3: Text processing function
# ================================================================================
def preprocess_text(text):
   text = text.lower() # Convert text to lowercase
   text = re.sub(r"[^a-zA-Z\s]", "", text) # Remove special chars/numbers
   stop_words = set(stopwords.words('english'))
   words = text.split()
   word = [w for w in words if w not in stop_words] # Remove stopwords
   lemmatizer = WordNetLemmatizer()
   word = [lemmatizer.lemmatize(w) for w in word]
   return ' '.join(word)

# Clean the text data by preprocessing it
texts_clean = [preprocess_text(text) for text in texts]

# Feature extraction with unigrams + bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2)) # Unigrams + Bigrams
X = vectorizer.fit_transform(texts_clean)

# ================================================================================
# SECTION 0: Train the model and make prediction
# ================================================================================
# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, labels,test_size=0.2, random_state=42,stratify=labels)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Cross-validation (5-fold)
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Cross-validation accuracy: ", scores.mean())

# ================================================================================
# SECTION 5: Data Visualisation Section 📊💹
# ================================================================================
# a) Confusion Matrix
cm = confusion_matrix(y_test, y_pred,labels=["positive","negative"])
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["positive","negative"], yticklabels=["positive","negative"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# b) Top Informative Features per Class
def plot_top_features(classifier, vectorizer, class_labels, n=10):
   feature_names = np.array(vectorizer.get_feature_names_out())
   for i, class_label in enumerate(class_labels):
      # Get top n features for each class
      top = np.argsort(classifier.coef_[i])[-n:]
      plt.figure(figsize=(12, 8))
      plt.barh(range(n),classifier.coef_[i][top], color="skyblue")
      plt.yticks(np.arange(n), feature_names[top])
      plt.title(f"Top {n} features for '{class_label}' Class")
      plt.xlabel("Log Probability")
      plt.tight_layout()
      plt.show()

# Re-train the model on full data for feature visualisation
model.fit(X,labels)
# model.coef_ = np.log(model.feature_log_prob_) # Convert to log probabilities for visualisation
model.coef_ = model.feature_log_prob_
plot_top_features(model, vectorizer, class_labels=["negative","positive"], n=10)

# c) Cross-validation Score distribution
plt.figure(figsize=(12, 8))
sns.boxplot(data=scores, orient='h', color="lightgreen")
plt.title("Cross-Validation Accuracy Scores (5-fold)")
plt.xlabel("Accuracy")
plt.tight_layout()
plt.show()