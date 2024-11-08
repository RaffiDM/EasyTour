# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load model dan data
@lru_cache(maxsize=None)
def load_models():
    model = tf.keras.models.load_model('model/tourism_classifier.h5')
    tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.joblib')
    scaler = joblib.load('model/scaler.joblib')
    label_encoder = joblib.load('model/label_encoder.joblib')
    tourism_df = pd.read_csv('data/tourism_with_id.csv')
    return model, tfidf_vectorizer, scaler, label_encoder, tourism_df

def prepare_place_features(tourism_df):
    """
    Menyiapkan fitur-fitur untuk perhitungan similarity
    """
    # Text features
    tourism_df['text_features'] = tourism_df.apply(
        lambda x: f"{x['Place_Name']} {x['Description']} {x['City']}", axis=1
    )
    
    # Numeric features - handle duplicate values
    try:
        tourism_df['price_category'] = pd.qcut(
            tourism_df['Price'], 
            q=5, 
            labels=['very_cheap', 'cheap', 'moderate', 'expensive', 'very_expensive'],
            duplicates='drop'  # Handle duplicate values
        )
    except ValueError:
        # If qcut fails due to too many duplicate values, use cut instead
        # Calculate custom bins based on data distribution
        price_min = tourism_df['Price'].min()
        price_max = tourism_df['Price'].max()
        # Create custom bins with more spread to handle skewed data
        bins = [
            price_min,
            price_min + (price_max - price_min) * 0.1,
            price_min + (price_max - price_min) * 0.25,
            price_min + (price_max - price_min) * 0.5,
            price_min + (price_max - price_min) * 0.75,
            price_max
        ]
        tourism_df['price_category'] = pd.cut(
            tourism_df['Price'],
            bins=bins,
            labels=['very_cheap', 'cheap', 'moderate', 'expensive', 'very_expensive'],
            include_lowest=True
        )
    
    return tourism_df

def predict_category(model, tfidf_vectorizer, scaler, label_encoder, place_name):
    """
    Memprediksi kategori tempat wisata
    """
    name_word_count = len(place_name.split())
    combined_features = f"{place_name} {name_word_count}"
    
    input_vector = tfidf_vectorizer.transform([combined_features]).toarray()
    input_scaled = scaler.transform(input_vector)
    prediction = model.predict(input_scaled)
    
    probabilities = prediction[0]
    predicted_category_index = np.argmax(probabilities)
    predicted_category = label_encoder.inverse_transform([predicted_category_index])[0]
    
    categories = label_encoder.classes_
    category_probs = {cat: float(prob) for cat, prob in zip(categories, probabilities)}
    
    return predicted_category, category_probs

def _generate_similarity_explanation(place, predicted_category, input_place=None):
    """
    Generates human-readable explanation for why this place was recommended
    """
    reasons = []
    
    # Content similarity
    if place['similarity_score'] > 0.5:
        reasons.append("memiliki deskripsi dan karakteristik yang sangat mirip")
    elif place['similarity_score'] > 0.3:
        reasons.append("memiliki beberapa kesamaan dalam deskripsi")
    
    # Category match
    if place['Category'] == predicted_category:
        reasons.append(f"termasuk dalam kategori yang sama ({predicted_category})")
    
    # Price similarity
    if input_place is not None:
        input_price = input_place['price_category'].iloc[0]
        if place['price_category'] == input_price:
            reasons.append("memiliki rentang harga yang sama")
        elif abs(ord(str(place['price_category'][0])) - ord(str(input_price)[0])) == 1:
            reasons.append("memiliki rentang harga yang berdekatan")
    
    if not reasons:
        reasons.append("memiliki beberapa kemiripan umum")
    
    return " dan ".join(reasons)

def get_recommendations(place_name, tourism_df, model, tfidf_vectorizer, scaler, label_encoder, n_recommendations=15):
    """
    Mendapatkan rekomendasi tempat wisata berdasarkan similarity konten
    """
    # Prepare features
    df = prepare_place_features(tourism_df.copy())
    
    # Get predicted category
    predicted_category, category_probs = predict_category(
        model=model,
        tfidf_vectorizer=tfidf_vectorizer,
        scaler=scaler,
        label_encoder=label_encoder,
        place_name=place_name
    )
    
    # Find input place
    input_place = df[df['Place_Name'].str.lower() == place_name.lower()]
    if input_place.empty:
        print(f"Warning: '{place_name}' tidak ditemukan dalam database.")
        input_text = place_name
    else:
        input_text = input_place['text_features'].iloc[0]
    
    # Calculate TF-IDF for all places
    content_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = content_vectorizer.fit_transform(df['text_features'])
    
    # Calculate similarity scores
    if input_place.empty:
        input_vector = content_vectorizer.transform([input_text])
        similarity_scores = cosine_similarity(input_vector, tfidf_matrix)[0]
    else:
        input_idx = input_place.index[0]
        similarity_scores = cosine_similarity(tfidf_matrix[input_idx:input_idx+1], tfidf_matrix)[0]
    
    # Add similarity scores to dataframe
    df['similarity_score'] = similarity_scores
    
    # Calculate category bonus
    df['category_bonus'] = df['Category'].apply(
        lambda x: 0.3 if x == predicted_category else 0
    )
    
    # Calculate price similarity bonus
    if not input_place.empty:
        input_price_category = input_place['price_category'].iloc[0]
        df['price_bonus'] = df['price_category'].apply(
            lambda x: 0.2 if x == input_price_category else 
                      0.1 if abs(ord(str(x)[0]) - ord(str(input_price_category)[0])) == 1 else 0
        )
    else:
        df['price_bonus'] = 0
    
    # Calculate final score
    df['final_score'] = df['similarity_score'] + df['category_bonus'] + df['price_bonus']
    
    # Filter out input place if it exists
    if not input_place.empty:
        df = df[df['Place_Name'] != place_name]
    
    # Sort by final score and get recommendations
    recommendations = df.nlargest(n_recommendations, 'final_score')
    
    # Prepare detailed recommendations with similarity explanation
    detailed_recommendations = []
    for _, row in recommendations.iterrows():
        similarity_details = {
            'name': row['Place_Name'],
            'city': row['City'],
            'category': row['Category'],
            'price': f"Rp {row['Price']:,}",
            'description': row['Description'][:200] + '...' if len(row['Description']) > 200 else row['Description'],
            'similarity_score': float(row['similarity_score']),
            'category_match': row['Category'] == predicted_category,
            'price_category': row['price_category'],
            'final_score': float(row['final_score']),
            'explanation': _generate_similarity_explanation(
                row, 
                predicted_category,
                input_place if not input_place.empty else None
            )
        }
        detailed_recommendations.append(similarity_details)
    
    return detailed_recommendations, predicted_category, category_probs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        place_name = request.form['place_name']
        model, tfidf_vectorizer, scaler, label_encoder, tourism_df = load_models()
        
        recommendations, category, category_probs = get_recommendations(
            place_name=place_name,
            tourism_df=tourism_df,
            model=model,
            tfidf_vectorizer=tfidf_vectorizer,
            scaler=scaler,
            label_encoder=label_encoder
        )
        
        return jsonify({
            'success': True,
            'predicted_category': category,
            'category_probabilities': category_probs,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)