"""
================================================================================
Python script to demonstrate a simple holiday recommendation system
================================================================================
The system recommends holidays destinations using collaborative filtering based
on user similarity as well as content based approach.

Requirements
--------------------------
 pip install matplotlib numpy pandas scikit-learn

Author : Nyanjui
Date   : 01 October 2026
"""
# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================================================================================
# SECTION 1: Create a dummy dataset
# ================================================================================
# Dummy dataset of users, holiday destinations, and ratings.
# Add a category for each destination (e.g., beach, city, mountain and so on) for
# content based filtering
data = {
    'user_id': [
        1,1,2,2,3,3,4,4,5,5,
        6,6,7,7,8,8,9,9,10,10,
        11,11,12,12,13,13,14,14,15,15,
        16,16,17,17,18,18,19,19,20,20,
        21,21,22,22,23,23,24,24,25,25,
        26,26,27,27,28,28,29,29,30,30
    ],

    'destination': [
        'Maldives','Paris','Maldives','Tokyo','Swiss Alps','Tokyo',
        'Maldives','Swiss Alps','Paris','Tokyo',
        'Hawaii','New York','Barcelona','Maldives','Swiss Alps','Kyoto',
        'Bali','Rome','New York','Bali',
        'Barcelona','Swiss Alps','Tokyo','Maldives','Rome','New York',
        'Bali','Paris','Hawaii','Kyoto',
        'Swiss Alps','Barcelona','Paris','Rome','Tokyo','Kyoto',
        'Maldives','New York','Bali','Hawaii',
        'Cape Town','Zanzibar',
        'Maasai Mara','Dubai',
        'Marrakech','Paris',
        'Victoria Falls','Cape Town',
        'Seychelles','Maldives',
        'Nairobi','Zanzibar',
        'Cairo','Rome',
        'Serengeti','Maasai Mara',
        'Lagos','Dubai',
        'Addis Ababa','Nairobi'
    ],

    'rating': [
        5,4,5,3,4,2,4,5,3,4,
        5,4,4,5,5,3,
        5,4,4,5,
        5,5,2,4,4,5,
        3,4,5,3,
        5,4,4,4,2,4,
        5,4,4,5,
        5,5,
        5,4,
        4,4,
        5,5,
        5,5,
        4,5,
        4,4,
        5,5,
        3,4,
        4,4
    ],

    'category': [
        'beach','city','beach','city','mountain','city',
        'beach','mountain','city','city',
        'beach','city','city','beach','mountain','city',
        'beach','city','city','beach',
        'city','mountain','city','beach','city','city',
        'beach','city','beach','city',
        'mountain','city','city','city','city','city',
        'beach','city','beach','beach',
        'city','beach',          # Cape Town, Zanzibar
        'wildlife','city',      # Maasai Mara, Dubai
        'city','city',          # Marrakech, Paris
        'nature','city',        # Victoria Falls, Cape Town
        'beach','beach',        # Seychelles, Maldives
        'city','beach',         # Nairobi, Zanzibar
        'historical','city',    # Cairo, Rome
        'wildlife','wildlife',  # Serengeti, Maasai Mara
        'city','city',          # Lagos, Dubai
        'city','city'           # Addis Ababa, Nairobi
    ]
}

# load above data into a Pandas DataFrame
df = pd.DataFrame(data)

# ================================================================================
# SECTION 2: Collaborative Filtering
# ================================================================================
# Create a pivot table with users as rows and destinations as columns, filled with ratings
pivot_table = df.pivot_table(index='user_id', columns='destination', values='rating').fillna(0)

# Calculate cosine similarity between users based on ratings
user_similarity = cosine_similarity(pivot_table)
user_similarity_df = pd.DataFrame(user_similarity, index=pivot_table.index, columns=pivot_table.index)

# Function to get recommendations for a specific user using collaborative filtering
def collaborative_recommend(user_id, num_recommendations=3):
   """
         Generate personalised holiday destination recommendations for a user
         using a collaborative filtering approach.

         This function identifies users with similar taste based on ratings,
         aggregates their ratings for destinations the target user has not yet
         visited, and returns the top destinations with the highest combined scores.

         Parameters
         ----------
         user_id : int
             The unique identifier of the user for whom recommendations are generated.
         num_recommendations : int, optional, default=3
             The number of recommended destinations to return.

         Returns
         -------
         list of str
             A list of recommended destination names, sorted by predicted preference
             in descending order.

         Notes
         -----
         - The function assumes `user_similarity_df` contains pairwise user similarity scores.
         - The function assumes `pivot_table` contains user ratings for each destination,
           with 0 indicating no rating.
         - Recommendations are based solely on ratings from similar users and do not
           consider other factors such as price, location, or category.
         """
   similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
   recommendations = {}

   # Collect ratings from similar users
   for similar_user in similar_users:
      for destination, rating in pivot_table.loc[similar_user].items():
         if pivot_table.loc[user_id, destination] == 0 and rating >0:
            if destination not in recommendations:
               recommendations[destination] = rating
            else:
               recommendations[destination] += rating

   # Sort and select top recommended destinations
   sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
   return [dest[0] for dest in sorted_recommendations[:num_recommendations]]

# Example: Get recommendations for user id 4 and 28
print(f"Collaborative filtering recommendations for: "
      f"\nuser id 4 -> {collaborative_recommend(4)}"
      f"\nuser id 28 -> {collaborative_recommend(28)}")

# ================================================================================
# SECTION 3: Content-based Filtering
# ================================================================================
# First, create a dataframe with unique destinations and categories
unique_destinations = df[['destination','category']].drop_duplicates().reset_index(drop=True)

# Create a content matrix using the category feature
count_vectorizer = CountVectorizer()
content_matrix = count_vectorizer.fit_transform(unique_destinations['category'])

# Calculate cosine similarity between destinations base on content
destination_similarity = cosine_similarity(content_matrix)
destination_similarity_df = pd.DataFrame(destination_similarity, index=unique_destinations['destination'],
                                         columns=unique_destinations['destination'])

# Function to get content-based recommendations based on a destinatin
def content_based_recommend(destination, num_of_recommendations=3):
   """
         Generate a list of holiday destinations similar to a given destination
         using a content-based filtering approach.

         This function identifies destinations that are most similar to the
         specified destination based on precomputed similarity scores,
         and returns the top matches.

         Parameters
         ----------
         destination : str
             The name of the destination for which similar destinations are recommended.
         num_recommendations : int, optional, default=3
             The number of similar destinations to return.

         Returns
         -------
         list of str
             A list of destination names most similar to the input destination,
             sorted by similarity in descending order.

         Notes
         -----
         - The function assumes `destination_similarity_df` contains pairwise similarity scores
           between destinations.
         - The input destination itself is excluded from the recommendations.
         - Recommendations are based solely on content similarity and do not take user preferences
           into account.
         """
   similar_destinations = destination_similarity_df[destination].sort_values(
      ascending=False).index[1:num_of_recommendations + 1]
   return similar_destinations.tolist()

# Example: Get recommendations for a destination like 'Nairobi' and 'Maldives'
print(f"\nContent Based Recommendations for: "
      f"\n'Nairobi' -> {content_based_recommend('Nairobi')}"
      f"\n'Maldives' -> {content_based_recommend('Maldives')}'")

# ================================================================================
# SECTION 4: Visualisation
# ================================================================================
# Show the similarity heatmap
plt.figure(figsize=(8,6))
plt.imshow(user_similarity, cmap='viridis', interpolation='none')
plt.colorbar(label='Similarity')
plt.xticks(range(len(pivot_table.index)), pivot_table.index)
plt.yticks(range(len(pivot_table.index)), pivot_table.index)
plt.title("User Similarity Heatmap")
plt.xlabel("User ID")
plt.ylabel("User ID")
plt.show()

# Show destination similarity heatmap (content-based)
plt.figure(figsize=(8,6))
plt.imshow(destination_similarity, cmap='viridis', interpolation='none')
plt.colorbar(label='Similarity')
plt.xticks(range(len(destination_similarity_df.columns)),destination_similarity_df.columns,rotation=90)
plt.yticks(range(len(destination_similarity_df.index)), destination_similarity_df.index)
plt.xlabel("Destination")
plt.ylabel("Destination")
plt.show()


