# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import csv

# Create Flask app
app = Flask(__name__)


# Load Dataset
def load_data(file_path):
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)


# Calculate distance between two geographical points
def calculate_distance(user_coords, spot_coords):
    print(f"Calculating distance between {user_coords} and {spot_coords}")
    return geodesic(user_coords, spot_coords).kilometers


# Collaborative Filtering
def collaborative_filtering(user_data, place_id, user_id=None):
    try:
        ratings_matrix = user_data.pivot_table(index='user_id',
                                               columns='place_id',
                                               values='rating').fillna(0)
        similarity_matrix = pd.DataFrame(cosine_similarity(ratings_matrix.T),
                                         index=ratings_matrix.columns,
                                         columns=ratings_matrix.columns)

        if place_id not in similarity_matrix.columns:
            print(f"place_id {place_id} not found in similarity matrix")
            return []

        similarity_scores = similarity_matrix[place_id].sort_values(
            ascending=False)
        return similarity_scores.tolist()
    except Exception as e:
        print(f"Error in collaborative_filtering: {e}")
        return []


# Filter valid tourist spots based on user input
def filter_tourist_spots(data,
                         hidden_gem=True,
                         is_valid=True,
                         current_day=None,
                         spot_type=None,
                         name=None,
                         ratings=None):
    print("Filtering tourist spots")
    filtered_data = data

    if name:
        filtered_data = filtered_data[filtered_data['name'].str.contains(
            name, case=False, na=False)]

    if spot_type and len(spot_type) > 0:
        result_string = '|'.join(spot_type)
        filtered_data = filtered_data[filtered_data['query'].str.contains(
            result_string, case=False, na=False)]

    if ratings:
        filtered_data = filtered_data[(filtered_data['rating'] >= ratings[0]) &
                                      (filtered_data['rating'] <= ratings[1])]

    return filtered_data


# Recommend tourist spots
def recommend_tourist_spots(data,
                            user_location,
                            similar_places,
                            n_recommendations=100):
    print("Recommending tourist spots")
    matching_data = data

    if similar_places:
        matching_data = data[data['place_id_y'].isin(similar_places)]
        matching_data['place_id_y'] = pd.Categorical(
            matching_data['place_id_y'],
            categories=similar_places,
            ordered=True)
        matching_data = matching_data.sort_values('place_id_y').reset_index(
            drop=True)

    matching_data = matching_data.sort_values('rating', ascending=False)
    return matching_data


@app.route('/recommend', methods=['POST'])
def recommend():
    print("Received recommendation request")
    user_longitude = request.json.get('longitude')
    user_latitude = request.json.get('latitude')
    hidden_gem = request.json.get('hidden_gem', True)
    current_day = datetime.now().strftime('%A')
    is_valid = request.json.get('is_valid', True)
    types = request.json.get('types')
    user_id = request.json.get('user_id')
    name = request.json.get('name')
    place_ids = request.json.get('place_ids', [])
    all = request.json.get('all', False)
    ratings = request.json.get('ratings')

    user_location = (
        user_latitude,
        user_longitude) if user_latitude and user_longitude else None
    file_path = 'dataset/scrapetable_wisata_cleaned.csv'
    data = load_data(file_path)
    file_path_user = 'dataset/output.csv'
    user_data = load_data(file_path_user)

    similar_places = []
    if not all and place_ids:
        combined_array = []
        for place_id in place_ids:
            similar_places = collaborative_filtering(user_data, place_id,
                                                     user_id)
            if isinstance(similar_places, pd.Series):
                similar_places = similar_places.index.tolist()
            combined_array.extend(similar_places)
        similar_places = list(set(combined_array))

    filtered_spots = filter_tourist_spots(data, hidden_gem, is_valid,
                                          current_day, types, name, ratings)
    recommendations = recommend_tourist_spots(filtered_spots,
                                              user_location,
                                              similar_places,
                                              n_recommendations=100)
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df['original_index'] = recommendations_df.index

    filtered_df = recommendations_df.groupby(
        'place_id_y',
        as_index=False).apply(lambda x: x[x['location_link'].notna()].head(1)
                              if not x[x['location_link'].notna()].empty else x
                              .head(1)).reset_index(drop=True)

    filtered_df = filtered_df.sort_values(by='original_index').drop(
        columns='original_index').reset_index(drop=True)
    return jsonify(filtered_df.to_dict(orient='records'))


@app.route('/add-review', methods=['POST'])
def add_review():
    print("Received add review request")
    user_id = request.json.get('user_id')
    place_id = request.json.get('place_id')
    name = request.json.get('name')
    rating = request.json.get('rating')

    if not all([user_id, place_id, name, rating]):
        return jsonify({"error": "Missing required parameters"}), 400

    review_data = {
        'user_id': user_id,
        'place_id': place_id,
        'name': name,
        'rating': rating
    }
    file_path = 'dataset/output.csv'

    try:
        with open(file_path, 'a') as file:
            writer = csv.DictWriter(file, fieldnames=review_data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(review_data)
        return jsonify({
            "message": "Review berhasil ditambahkan",
            "data": review_data
        }), 201
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500

@app.route('/keep-alive', methods=['GET'])
def keep_alive():
    return "I'm alive!", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
