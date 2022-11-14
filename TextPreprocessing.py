import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Model(object):

    def __init__(self):
        self.text = None
        self.language = None

    def preprocess(self, text):
        self.text = self.lowercase(text)
        self.text = self.textCleansing(self.text)
        self.text, self.language = self.detectLanguage(self.text)
        self.text = self.stopwordsRemove(self.text, self.language)

        if (self.language == 'en'):
            self.text = self.lemmatization(self.text)
        else:
            self.text = self.stemming(self.text)

        self.text = str(' '.join(self.text))

        return self.text, self.language

    def lowercase(self, text):
        return text.lower()

    def textCleansing(self, text):
        text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
        text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
        text = re.sub(r'www.[^ ]+', '', text)
        text = re.sub(r'[a-zA-Z0-9]*www[a-zA-Z0-9]*com[a-zA-Z0-9]*', '', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = [token for token in text.split() if len(token) > 2]
        text = ' '.join(text)
        return text

    def get_lang_detector(self, nlp, name):
        return LanguageDetector()

    def detectLanguage(self, text):
        nlp = spacy.load("en_core_web_sm")
        Language.factory("language_detector", func=self.get_lang_detector)
        nlp.add_pipe('language_detector', last=True)

        doc = nlp(text)
        language = doc._.language['language']
        return text, language

    def stopwordsRemove(self, text, language):
        if language == 'en':
            stop_words = set(stopwords.words('english'))
            stop_words.remove('not')
        else:
            stop_words = stopwords.words('indonesian')

        word_tokens = word_tokenize(text)
        filtered_text = [
            word for word in word_tokens if word not in stop_words]
        return filtered_text

    def lemmatization(self, text):
        lemmatizer = WordNetLemmatizer()

        lemmas = [lemmatizer.lemmatize(word=x, pos='v') for x in text]
        return lemmas

    def stemming(self, text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        stemmed_tokens = [stemmer.stem(token) for token in text]
        return stemmed_tokens
