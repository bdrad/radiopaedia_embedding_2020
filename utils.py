from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

LABELS = (
 'Obstetrics',
 'Paediatrics',
 'Musculoskeletal',
 'Gynaecology',
 'CNS',
 'Misc',
 'Hepatobiliary',
 'Oncology',
 'HN',
 'Breast',
 'Gastrointestinal',
 'Cardiac',
 'Chest',
 'Trauma',
 'Urogenital',
 'Vascular',
 'Spine',
 'Haematology',
 'Forensic',
 'Interventional'
)

def preprocess(text):
    stop_words = set(stopwords.words("english"))
    punctuation_set = set(i for i in string.punctuation)
    # make lowercase
    text = text.lower()
    # remove no width space
    text = text.replace('\u200b', '')
    # separate out punctuation
    for mark in punctuation_set:
        text = text.replace(mark, f' {mark} ')
    # tokenize using nltk punkt
    tokens = word_tokenize(text)
    #remove stopwords, punctuation, and standalone numerals
    tokens = [i for i in tokens \
        if i not in stop_words \
        if i not in punctuation_set \
        if not i.isnumeric()]
    
    return tokens
