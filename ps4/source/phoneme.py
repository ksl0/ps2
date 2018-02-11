"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2018 Feb 5
Description : Perceptron vs Logistic Regression on a Phoneme Dataset
"""

# utilities
from util import *

# scipy libraries
from scipy import stats

# scikit-learn libraries
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron, LogisticRegression

######################################################################
# functions
######################################################################

def cv_performance(clf, train_data, kfs) :
    """
    Determine classifier performance across multiple trials using cross-validation
    
    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kfs        -- array of size n_trials
                      each element is one fold from model_selection.KFold
    
    Returns
    --------------------
        scores     -- numpy array of shape (n_trials, n_fold)
                      each element is the (accuracy) score of one fold in one trial
    """
    
    n_trials = len(kfs)
    n_folds = kfs[0].n_splits
    scores = np.zeros((n_trials, n_folds))
    
    ### ========== TODO : START ========== ###
    # part b: run multiple trials of CV
    
    ### ========== TODO : END ========== ###
    
    return scores


def cv_performance_one_trial(clf, train_data, kf) :
    """
    Compute classifier performance across multiple folds using cross-validation
    
    Parameters
    --------------------
        clf        -- classifier
        train_data -- Data, training data
        kf         -- model_selection.KFold
    
    Returns
    --------------------
        scores     -- numpy array of shape (n_fold, )
                      each element is the (accuracy) score of one fold
    """
    
    scores = np.zeros(kf.n_splits)
    
    ### ========== TODO : START ========== ###
    # part b: run one trial of CV
    
    ### ========== TODO : END ========== ###
    
    return scores


######################################################################
# main
######################################################################

def main() :
    np.random.seed(1234)
    
    #========================================
    # load data
    train_data = load_data("phoneme_train.csv")
    
    ### ========== TODO : START ========== ###
    # part a: is data linearly separable?
    clf = Perceptron()
    clf.fit(train_data.X, train_data.y)
    print "coefs = %s, iteration = %s" % (clf.coef_, clf.n_iter_)
    ### ========== TODO : END ========== ###
        
    ### ========== TODO : START ========== ###
    # part c-d: compare classifiers
    # make sure to use same folds across all runs
    
    ### ========== TODO : END ========== ###
    
    ### ========== TODO : START ========== ###
    # part e: plot
    
    ### ========== TODO : END ========== ###

if __name__ == "__main__" :
    main()