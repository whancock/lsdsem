import nltk
from collections import Counter

class Story():

    """
    definitions:
        context: the body of the story, in this case, the first four sentences
        endings: the two possible endings for each story

    """

    def __init__(self, story_id, context, endings, ending_idx):

        self.story_id = story_id
        self.sentences = context + endings
        self.ending_idx = ending_idx

        self.tokenized_sentences = [self.process_sent(sent) for sent in self.sentences]
        self.counter = Counter(self.get_all_tokens())


    def __str__(self):
        return '{} {} {}'.format(self.story_id, self.sentences, self.ending_idx)


    def get_endings(self):
        return self.sentences[-2:]

    def get_context(self):
        return self.sentences[:4]

    def get_tokenized_endings(self):
        return self.tokenized_sentences[-2:]

    def get_tokenized_context(self):
        return self.tokenized_sentences[:4]



    def get_context_tokens(self):
        return [word for sent in self.get_tokenized_context() for word in sent]


    def get_all_tokens(self):
        return [word for sent in self.tokenized_sentences for word in sent]


    def get_unique_tokens(self):
        return set(self.counter.most_common())


    def get_freq(self):
        return self.counter


    def process_sent(self, sent):

        tokens = []

        words = nltk.word_tokenize(sent)

        for word in words:

            word = word.lower()
            # TODO: maybe some other processing here
            tokens.append(word)

        return tokens

