import pandas as pd
import re, nltk
from sklearn.feature_extraction.text import CountVectorizer        
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

import random
regex = "[^a-zA-Z]"
stemmer = PorterStemmer()
test_data_file_name = 'test_tweet.txt'
train_data_file_name = 'train_tweet.txt'

test_data_df = pd.read_csv(test_data_file_name, header=None, 
                                                     delimiter="\t", quoting=3)
test_data_df.columns = ["Text"]
train_data_df = pd.read_csv(train_data_file_name, header=None, delimiter="\t", 
                                                                     quoting=3)
train_data_df.columns = ["Sentiment","Text"]

classifier_dict = dict()


def best_classifier():
   print("==")
   print("\t\tClassifiers Accuracy List for 40% train data 60-40 split.")
   print("==")
   for key,value in sorted(classifier_dict.items(), key=lambda p:p[1], reverse=True):
       print(key,value,"%")
       print()

def print_the_sample_prediction():
    list1 = []
    for text, sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
        data = str(sentiment) + '\t' + text
        list1.append(data)
        list1.append('\n')
    print("==")
    print("\t\t\tSample obtained Result")
    print("1 implies positive sentiment and 0 implies otherwise.")
    print("==")
    print('Sentiment','\t\t','Text')
    print()
    
    for item in list1:
        print(item)

def cross_validation_report(y_test, y_pred, classifier):
    print()
    print("==")
    print("\t\t\t",classifier)
    print("==")
    print(classification_report(y_test, y_pred))
    print()

def cross_validation_score(score, classifier):
    print()
    print("==")
    print("\t\t\tCross Validation 4-folds")
    print("==")
    print("Cross Validation with 4 Folds")
    print("4 Fold Scores    : {}".format(score))
    print("Scores after Mean: ",score.mean())
    round_score = round(score.mean(), 4)
    classifier += "::"
    classifier_dict[classifier] = round_score * 100
    

def fit_model_and_predict(classifier_obj):
    classifier_obj = classifier_obj.fit(X=X_train, y=y_train)
    y_pred = classifier_obj.predict(X_test)
    return classifier_obj,y_pred

def split_the_data():
    return train_test_split(
        combined_array_data[0:len(train_data_df)], 
        train_data_df.Sentiment,
        train_size=0.60, 
        random_state=1234)

def fit_model():
    corpus_data_features = count_matrix.fit_transform(
    train_data_df.Text.tolist() + test_data_df.Text.tolist())
    
    return corpus_data_features

def create_count_vector_matrix():
    count_matrix = CountVectorizer(analyzer = 'word', tokenizer = create_tokens,
                               lowercase = True, stop_words = 'english',
                               max_features = None)
    return count_matrix
    

def do_stemming(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def create_tokens(text_senti):
    text_senti = re.sub(regex, " ", text_senti)
    tokens = nltk.word_tokenize(text_senti)
    stems = do_stemming(tokens, stemmer)
    return stems

count_matrix = create_count_vector_matrix()
combined_data = fit_model()
combined_array_data = combined_data.toarray()
print(combined_array_data.shape)

X_train, X_test, y_train, y_test = split_the_data()


# Logistic Regression
Logistic_classifier_obj = LogisticRegression()
Logistic_classifier_obj,y_pred = fit_model_and_predict(Logistic_classifier_obj)
score = cross_val_score(Logistic_classifier_obj, X_train, y_train, cv=4)
cross_validation_score(score,"Logistic Regression")
cross_validation_report(y_test, y_pred, "Logistic Regression")
# train classifier
logistic = LogisticRegression()
logistic = logistic.fit(X=combined_array_data[0:len(train_data_df)], 
                                                     y=train_data_df.Sentiment)
# get predictions
test_pred = logistic.predict(combined_array_data[len(train_data_df):])
# sample some of them
spl = random.sample(range(len(test_pred)), 5)
print_the_sample_prediction()

# Gaussian Naive Bayes
Naive_classifier_obj = GaussianNB()
Naive_classifier_obj,y_pred = fit_model_and_predict(Naive_classifier_obj)
score = cross_val_score(Naive_classifier_obj, X_train, y_train, cv=4)
cross_validation_score(score,"Gaussian Naive Bayes")
cross_validation_report(y_test, y_pred, "Gaussian Naive Bayes")
# train classifier
naive = GaussianNB()
naive = naive.fit(X=combined_array_data[0:len(train_data_df)], 
                                                     y=train_data_df.Sentiment)
# get predictions
test_pred = naive.predict(combined_array_data[len(train_data_df):])
# sample some of them
spl = random.sample(range(len(test_pred)), 500)
print_the_sample_prediction()





# Multinomial Naive Bayes
Multinomia_Naive_classifier_obj = MultinomialNB()
Multinomia_Naive_classifier_obj,y_pred = fit_model_and_predict(Multinomia_Naive_classifier_obj)
score = cross_val_score(Multinomia_Naive_classifier_obj, X_train, y_train, cv=4)
print()
cross_validation_score(score,"Multinomial Naive Bayes")
cross_validation_report(y_test, y_pred, "Multinomial Naive Bayes")
 # train classifier
multi_naive = MultinomialNB()
multi_naive = multi_naive.fit(X=combined_array_data[0:len(train_data_df)], 
                                              y=train_data_df.Sentiment)
# get predictions
test_pred = multi_naive.predict(combined_array_data[len(train_data_df):])
# sample some of them
spl = random.sample(range(len(test_pred)), 5)
print_the_sample_prediction()





 
# Decission Tree Regressor
Decision_classifier_obj = tree.DecisionTreeRegressor()
Decision_classifier_obj,y_pred = fit_model_and_predict(Decision_classifier_obj)
score = cross_val_score(Decision_classifier_obj, X_train, y_train, cv=4)
cross_validation_score(score,"Decision Tree Regressor") 
cross_validation_report(y_test, y_pred, "Decision Tree Regressor")
# train classifier
dtree = tree.DecisionTreeRegressor()
dtree = dtree.fit(X=combined_array_data[0:len(train_data_df)], 
                                                      y=train_data_df.Sentiment)
# get predictions
test_pred = dtree.predict(combined_array_data[len(train_data_df):])
spl = random.sample(range(len(test_pred)), 5)
print_the_sample_prediction()



# For ouptut
best_classifier()
