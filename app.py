from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load vectorizer dan model dari file
with open("tfidf.pkl", "rb") as vec_file:
    tfidf_vectorizer = pickle.load(vec_file)

with open("c45.pkl", "rb") as mod_file:
    sentiment_model = pickle.load(mod_file)

# Fungsi untuk mengubah prediksi menjadi label sentimen
def get_sentiment_label(pred):
    return 'Positif' if pred == 1 else 'Negatif'

# Route untuk halaman utama
@app.route('/')
def main_page():
    return render_template('index.html')

# Route untuk prediksi sentimen
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        input_text = request.form['text']
        text_vector = tfidf_vectorizer.transform([input_text])
        prediction = sentiment_model.predict(text_vector)[0]
        prediction_label = get_sentiment_label(prediction)
        return render_template('index.html', text=input_text, prediksi=prediction_label)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
