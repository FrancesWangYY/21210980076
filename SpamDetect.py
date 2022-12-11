from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer

def load_one_file(filename):
    x=""
    with open(filename, errors='ignore') as f:
        for line in f:
            line=line.strip('\n')
            line = line.strip('\r')
            x+=line
    return x
 
def load_files_from_dir(rootdir):
    x=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v=load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    ham=[]
    spam=[]
    for i in range(1,3): #(1,6)
        path="data/enron"+str(i)+"/ham/"
        print ("Load %s" %path)
        ham+=load_files_from_dir(path)
        path="data/enron"+str(i)+"/spam/"
        print ("Load %s" %path)
        spam+=load_files_from_dir(path)
    return ham, spam

def get_features_by_wordbag():
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print (vectorizer)
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    return x,y

def get_features_by_wordbag_tfidf():
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vectorizer = CountVectorizer(binary=False,
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(x)
    x = tfidf.toarray()
    return  x,y

def do_svm_wordbag(x_train, x_test, y_train, y_test):
    print ("SVM and wordbag")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print ("NB and wordbag")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))


def do_dnn_wordbag(x_train, x_test, y_train, y_testY):
    print ("DNN and wordbag")
 
    # Building deep neural network
    clf = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes = (8, 5),
                        random_state = 1)
    print (clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))


max_features=8000
max_document_length=100

x,y=get_features_by_wordbag()
# x,y = get_features_by_wordbag_tfidf()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# do_svm_wordbag(x_train, x_test, y_train, y_test)
# do_nb_wordbag(x_train, x_test, y_train, y_test)
do_dnn_wordbag(x_train, x_test, y_train, y_test)

"""wordbag
SVM and wordbag
0.9699057287889775
[[1959   31] 
 [  52  716]]
NB and wordbag
0.9423495286439448
[[1883  107]   
 [  52  716]]  
DNN and wordbag
0.9778825235678028
[[1951   39]
 [  22  746]]
"""

"""tfidf
SVM and wordbag
0.9836838288614939
[[1971   19] 
 [  26  742]]
NB and wordbag
0.9601160261058739
[[1955   35]
 [  75  693]]
DNN and wordbag
0.9822335025380711
[[1965   25] 
 [  24  744]]
"""
