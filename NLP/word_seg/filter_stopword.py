stopwords = ['是']

def filter_stopword(input):
    return [i for i in input if i not in stopwords]