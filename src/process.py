import csv
import re
import numpy as np
#from neural_network.cloze_stories.data import ClozeStoriesData
#from neural_network.cloze_stories.data.models import Dataset, Story, DataSplit


def get_sentence(text):
    """

    :param text: basestring
    :rtype: Sentence
    """
    text = re.sub('[^0-9a-zA-Z ]+', '', text)
    text = text.lower()

    tokens = text.split()
    return tuple((text, tokens))




def load_stories_sct_fmt(path):
    """

    Load data in the Story Cloze Test format:

    story_id    snt1    snt2    snt3    snt4    ending1     ending2     label   feature1    feature2    ...

    Note that label is either "1" or "2", so we subtract "1" to make it binary
    :rtype: stories (list)

    """

    stories = []

    with open(path, 'r') as csvfile:

        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        header = next(reader)

        print(header)

        num_elems = len(header)

        for row in list(reader):

            # print(row)

            story_id = row[0]
    
            sentences = []
            for text in row[1:5]:
                sentences.append(get_sentence(text))
    
            potential_endings = []
            for text in row[5:7]:
                potential_endings.append(get_sentence(text))

            correct_endings = [int(row[7]) - 1]
    
    
    
            feature_values1 = []
            feature_values2 = []
            if len(row) >= 9:
                for idx in range(8, num_elems):
                    value = row[idx]
                    if "E1" in header[idx]:
                        feature_values1.append(float(value))
                    elif "E2" in header[idx]:
                        feature_values2.append(float(value))
                    else:
                        feature_values1.append(float(value))
                        feature_values2.append(float(value))

            potential_endings[0].metadata['feature_values'] = feature_values1
            potential_endings[1].metadata['feature_values'] = feature_values2
            stories.append(tuple(sentences, potential_endings,
                                 correct_endings, id=story_id))

    return stories


load_stories_sct_fmt(
    '/home/william.hancock/workspace/ukp/data/Data_for_neural_network/train__rocstories_kde_with_features.csv')
