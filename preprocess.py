def preprocess(sentence):
    sentence = (sentence.translate(str.maketrans('', '',string.punctuation)))
    sentence=word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    sentence = [lemmatizer.lemmatize(sent) for sent in sentence]
    stop_words = set(stopwords.words("english"))
    sentence = [sent for sent in sentence if sent not in stop_words]
    return ' '.join(sentence)
