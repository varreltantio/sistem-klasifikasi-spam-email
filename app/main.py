from TextPreprocessing import Model
from flask import Flask, render_template, jsonify, request
import joblib
import imaplib
import email
import csv

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text')
def text():
    return render_template('text.html')

@app.route('/predict-text', methods=['POST'])
def predict_text():
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

        # Feature extraction
        result = tf_idf.transform([clean_message]).toarray()

        # Get prediction
        prediction = model.predict(result)[0]

        return jsonify({'prediction': str(prediction)})
    except ValueError:
        return jsonify({'error': ValueError})

    
@app.route('/gmail')
def gmail():
    return render_template('gmail.html')


@app.route('/predict-gmail', methods=['POST'])
def predict_gmail():
    try:
        json_ = request.get_json()

        user_email = str(json_['email'])
        user_password = str(json_['password'])

        data = []

        with imaplib.IMAP4_SSL(host="imap.gmail.com", port=imaplib.IMAP4_SSL_PORT) as imap_ssl:
            print("Connection Object : {}".format(imap_ssl))

            # Login to Mailbox
            print("Logging into mailbox...")
            resp_code, response = imap_ssl.login(
                user_email, user_password)

            print("Response Code : {}".format(resp_code))
            print("Response      : {}\n".format(response[0].decode()))

            # Set Mailbox
            resp_code, mail_count = imap_ssl.select(mailbox="ELITMUS", readonly=True)
            imap_ssl.select()

            # Retrieve Mail IDs for given Directory
            resp_code, mail_ids = imap_ssl.search(None, "ALL")

            print("Mail IDs : {}\n".format(mail_ids[0].decode().split()))

            # Display Few Messages for given Directory
            for mail_id in mail_ids[0].decode().split()[-5:]:
                # Fetch mail data.
                resp_code, mail_data = imap_ssl.fetch(mail_id, '(RFC822)')

                # Construct Message from mail data
                message = email.message_from_bytes(mail_data[0][1])

                text = ""
    
                for part in message.walk():
                    if part.get_content_type() == "text/plain":
                        body_lines = part.as_string().split("\n")
                        text = "".join(body_lines[3:12])

                if text == "":
                    continue

                preprocess_model = Model()
                clean_message, language = preprocess_model.preprocess(text)

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

                # Feature extraction
                result = tf_idf.transform([clean_message]).toarray()

                # Get prediction
                prediction = model.predict(result)[0]

                if prediction == 0:
                    prediction = 'bukan spam'
                elif prediction == 1 or prediction == 2:
                    prediction = 'spam'

                data.append([message.get("From"), message.get("Subject"), message.get("Date"), text, prediction])

            # Close Selected Mailbox
            imap_ssl.close()


        header = ['from', 'subject', 'date', 'message', 'label']
        with open('dataset/gmail-inbox.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)

        
        return jsonify({'prediction': data})
    except ValueError:
        return jsonify({'error': ValueError})

@app.route('/result')
def result():
    with open("dataset/gmail-inbox.csv") as file:
        reader = csv.reader(file)
        header = next(reader)
        return render_template('result.html', header=header, rows=reader)