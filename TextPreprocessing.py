import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Model(object):

    def __init__(self, text):
        self.text = text

        self.text = self.lowercase(self.text)
        self.text = self.preprocess_remove(self.text)
        self.text = self.stopwordsRemove(self.text)
        self.text = self.lemmatization(self.text)

        self.text = str(' '.join(self.text))

    def lowercase(self, message):
        return message.lower()

    def preprocess_remove(self, text):
        text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
        text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
        text = re.sub(r'www.[^ ]+', '', text)
        text = re.sub(r'[a-zA-Z0-9]*www[a-zA-Z0-9]*com[a-zA-Z0-9]*', '', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = [token for token in text.split() if len(token) > 2]
        text = ' '.join(text)
        return text

    def stopwordsRemove(self, inputs):
        stop_words = set(stopwords.words('english'))
        stop_words.remove('not')

        word_tokens = word_tokenize(inputs)

        filtered_text = [
            word for word in word_tokens if word not in stop_words]
        return filtered_text

    def lemmatization(self, inputs):
        lemmatizer = WordNetLemmatizer()

        lemmas = [lemmatizer.lemmatize(word=x, pos='v') for x in inputs]
        return lemmas
