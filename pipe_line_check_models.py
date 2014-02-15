'''
Good code to run multiple models

'''

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from  sklearn.linear_model import  LogisticRegression, LinearRegression
import pdb
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn.svm import SVC

print "loading data.."
traindata = list(np.array(p.read_table('train.tsv'))[:,2])
traindata_label = np.array(p.read_table('train.tsv'))[:,-1]

def evaluate_cross_validation(classifier, train, labels, K):
    
    cv = KFold(len(labels), K, shuffle=True, random_state=0)
    scores =  cross_val_score(classifier, train, labels, cv=cv, scoring='roc_auc')
    print scores
    print ("Mean Cross Validation score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


# classifier pipeline

clf_1 = Pipeline([
    ('vect', TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
             analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,      
    )),
    ('clf', LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None,    
    )),
])
clf_2 = Pipeline([
    ('vect', TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
             analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,      
    )),
    ('clf', LogisticRegression()),    
])

clf_3 = Pipeline([
    ('vect', TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
             analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,      
    )),
    ('svc', SVC(probability=True)),
])

clfs = [clf_1, clf_2,clf_3]
for clf in clfs:
  print 'running model' 
  evaluate_cross_validation(clf, traindata, traindata_label, 5)5)