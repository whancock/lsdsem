import nltk

from collections import Counter

class Story():

    def __init__(self, id, sentences, endings, ending_idx):

        self.id = id
        self.sentences = sentences
        self.endings = endings
        self.ending_idx = ending_idx


    def __str__(self):
        return '{} {} {} {}'.format(self.id, self.sentences, self.endings, self.ending_idx)


    def get_tokens(self):

        tokens = Counter()

        for sent in self.sentences:
            tokens.update(self.process_sent(sent))
        
        for sent in self.endings:
            tokens.update(self.process_sent(sent))

        return tokens



    def process_sent(self, sent):

        tokens = []

        words = nltk.word_tokenize(sent)

        for word in words:

            word = word.lower()
            # TODO: maybe some other processing here
            tokens.append(word)

        return tokens



    def is_number(self, s):

        try:
            float(s)
            return True
        except ValueError:
            return False