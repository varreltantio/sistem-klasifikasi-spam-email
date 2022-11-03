import TextPreprocessing
from flask import Flask, render_template, jsonify, request
import joblib

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.get_json()

        message = str(json_['message'])
        clean_message = TextPreprocessing.Model(message)

        # TF - IDF
        tf_idf = joblib.load("tf_idf.save")

        # Naive Bayes Classifier Model
        model = joblib.load("nb.pkl")

        # Get prediction
        data = tf_idf.transform([clean_message.text]).toarray()

        prediction = model.predict(data)[0]

        return jsonify({'prediction': str(prediction)})
    except ValueError:
        return jsonify({'error': ValueError})
