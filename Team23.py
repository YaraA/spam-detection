import nltk
import sklearn
import scipy.stats
import os
import pyphen
import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV
from sklearn_crfsuite import scorers
from nltk import pos_tag, word_tokenize, sent_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


"""
Part 1: Reading and Preprocessing Data
"""

"""
Reads the emails from dataset.
Classify the email as a spam or not (1/0) according to the file name and
returns the email's contents and the email's class.
"""
def read_mails():
    mails_targets = []
    mails_data = []
    for root, dirs, _ in os.walk('bare/'):
        for sub_dir in dirs:
            for root, dirs, files in os.walk('bare/' + sub_dir):
                for file in files:
                    f = open(os.path.join('bare/' + sub_dir, file), 'r')
                    mails_data.append(f.read())
                    if "spm" in file:
                        mails_targets.append(1)
                    else:
                        mails_targets.append(0)
                    f.close()
    return mails_data, mails_targets

"""
Part 2: Scikit-Learn Classifiers
"""

"""
Takes the classifier, vectorizer, training data, test data, test targets and type of classifier.
Feeds a classifier using the training data and labels then
predicts on the testing data whether an email is ham or spam then
compars the predictions made with the testing labels and
returns the precision, recall and fscore of the classifier on the testing data.
"""
def metrics_classifier(model, vectorizer, train_mails, test_mails, test_targets, model_name):
    model.fit(train_mails, train_targets)
    transformed_test = vectorizer.transform(test_mails)
    predictions = model.predict(transformed_test)
    (precision, recall, fscore, _) = metrics.precision_recall_fscore_support(test_targets, predictions, average='macro')
    print(model_name)
    print("Precision:",precision, "Recall:", recall, "Fscore:", fscore)
    return precision, recall, fscore

"""
Fits and transforms the training data using the vectorizer.
Intializes and finds the metrics of the three classifiers:
a) Multinomial Naive Bayes
b) K Neighbors Classifier
c) Random Forest Classifier
"""
def classification(v, train_mails, train_targets, test_mails, test_targets):
    train_mails = v.fit_transform(train_mails)

    nb_clf = MultinomialNB()
    precision, recall, fscore = metrics_classifier(nb_clf, v, train_mails, test_mails, test_targets, "Multinomial Naive Bayes")

    kn_clf = KNeighborsClassifier()
    precision, recall, fscore = metrics_classifier(kn_clf, v, train_mails, test_mails, test_targets, "K Neighbors Classifier")

    rf_clf = RandomForestClassifier(random_state=0)
    precision, recall, fscore = metrics_classifier(rf_clf, v, train_mails, test_mails, test_targets, "Random Forest Classifier")

"""
Part 3: Classifying using Readability Features
"""

"""
Reads the spam phrases from the spam list.
"""
def read_spams_phrases():
    spam_words = []
    with open('spam_words.txt', 'r') as f:
        spam_words = f.read()
    return spam_words.split('\n')[:-1]

"""
Counts the number of occurances of each spam phrase in an email.
"""
def count_spam_phrases(email):
    count = 0
    for spam in spam_phrases:
        count += email.count(spam)
    return count

"""
Counts the number of verbs in an email.
"""
def count_verbs(email):
    for sentence in sent_tokenize(email):
        word_tags = pos_tag(word_tokenize(sentence))
        count = 0
        for word in word_tags:
            if word[1] == 'VBZ':
                count += 1
    return count

"""
Checks if a word contains both digits and alphabets.
"""
def contains_num_alpha(word):
    find_num = find_alpha = False
    for c in word:
        find_num = find_num or c.isdigit()
        find_alpha = find_alpha or c.isalpha()
    return find_num and find_alpha

"""
Counts the number of words containing both digits and alphabets in an email.
"""
def count_num_alpha(email):
    count = 0
    for word in word_tokenize(email):
        if contains_num_alpha(word):
            count += 1
    return count

"""
Counts the number of words containing 3 syllables in an email.
"""
def count_3_syllables(email):
    count = 0
    dic = pyphen.Pyphen(lang='en_GB')
    for word in word_tokenize(email):
        syllables_in_word = len(dic.inserted(word).split('-'))
        if syllables_in_word >= 3:
            count += 1
    return count

"""
Finds the average number of syllables in all words in an email.
"""
def avg_syllables(email):
    syllables_in_words = 0
    dic = pyphen.Pyphen(lang='en_GB')
    words = word_tokenize(email)
    for word in words:
        syllables_in_words += len(dic.inserted(word).split('-'))
    avg = syllables_in_words // len(words)
    return avg

"""
Extracts features from each email.
F1: The number of sentences in an email.
F2: The number of verbs in an email.
F3: The number of words containing both numeric and alphabetical characters.
F4: The number of words in an email that are found in the spam list.
F5: The number of words in an email that have more than 3 syllables.
F6: The average number of syllables of words in an email.
"""
def email2features(email):
    features = {
        'sents_count': len(sent_tokenize(email)),
        'verbs_count': count_verbs(email),
        'num_alpha_count': count_num_alpha(email),
        'spam_sentences_count': count_spam_phrases(email),
        '3_syllables_count': count_3_syllables(email),
        'avg_syllables': avg_syllables(email)
    }
    return features

"""
Builds a feature matrix (list of dicts), where every row corresponds to an email, and every column corresponds to a feature value of this email.
"""
def build_features_matrix(emails):
    matrix = []
    for email in emails:
        features = email2features(email)
        matrix.append(features)
    return matrix


mails_data, mails_targets = read_mails()
split = int(0.8 * len(mails_data))
train_mails = mails_data[:split]
train_targets = mails_targets[:split]
test_mails = mails_data[split:]
test_targets = mails_targets[split:]

spam_phrases = read_spams_phrases()

cv = CountVectorizer()
dv = DictVectorizer(sparse=False)
train_features = build_features_matrix(train_mails)
test_features = build_features_matrix(test_mails)

print("Classifying using email's whole text content")
classification(cv, train_mails, train_targets, test_mails, test_targets)
print("Classifying using Readability Features")
classification(dv, train_features, train_targets, test_features, test_targets)
