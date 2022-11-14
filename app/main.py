from TextPreprocessing import Model
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
        preprocess_model = Model()
        clean_message, language = preprocess_model.preprocess(message)

        if (language == 'en'):
            # TF - IDF
            tf_idf = joblib.load("model/tf_idf_en.save")

            # Naive Bayes Classifier Model
            model = joblib.load("model/nb_en.pkl")
        else:
            # TF - IDF
            tf_idf = joblib.load("model/tf_idf_id.save")

            # Naive Bayes Classifier Model
            model = joblib.load("model/nb_id.pkl")

        # Get prediction
        data = tf_idf.transform([clean_message]).toarray()

        prediction = model.predict(data)[0]

        return jsonify({'prediction': str(prediction)})
    except ValueError:
        return jsonify({'error': ValueError})
