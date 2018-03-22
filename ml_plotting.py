## Plotting utilities
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, title, X, y, ylim=(0,1), cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '--', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_various_max_leaf(title, X, y, ylim=(0,1), cv=None, step_size= 10, min_nodes=2, max_nodes= 10):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Max Number of Leaf Nodes")
    plt.ylabel("Score")
                          
    train_nodes = np.linspace(min_nodes,max_nodes,step_size)
    train_scores = np.zeros([step_size,cv])
    test_scores = np.zeros([step_size,cv])
    i = 0
    
    for num_node in train_nodes:
        clf = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes = int(num_node))
        clf.fit(X,y)
        cv_scores = cross_validate(clf,X,y, cv=cv)
        train_scores[i] = cv_scores.get('train_score')
        test_scores[i] = cv_scores.get('test_score')
        i += 1
        
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_nodes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_nodes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_nodes, train_scores_mean, '--', color="r",
             label="Training score")
    plt.plot(train_nodes, test_scores_mean, '--', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_various_neigbors(title, X, y, weights ='uniform', ylim=(0,1), cv=None, step_size= 10, max_neighbors= 10):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Neighbors")
    plt.ylabel("Score")
                          
    neighbors = np.linspace(1,max_neighbors,step_size)
    train_scores = np.zeros([step_size,cv])
    test_scores = np.zeros([step_size,cv])
    i = 0
    
    for n in neighbors:
        print("neighbord:", i)
        clf = KNeighborsClassifier(n_neighbors = int(n), weights = weights)
        clf.fit(X,y)
        cv_scores = cross_validate(clf,X,y, cv=cv)
        train_scores[i] = cv_scores.get('train_score')
        test_scores[i] = cv_scores.get('test_score')
        i += 1

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(neighbors, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(neighbors, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(neighbors, train_scores_mean, '--', color="r",
             label="Training score")
    plt.plot(neighbors, test_scores_mean, '--', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_compare_knn_weights(title, X, y, ylim=(0,1), cv=None, step_size= 10, max_neighbors= 10):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Neighbors")
    plt.ylabel("Score")
                          
    neighbors = np.linspace(1,max_neighbors,step_size)
    uniform_scores = np.zeros([step_size,cv])
    weighted_scores = np.zeros([step_size,cv])
    i = 0
    
    for n in neighbors:
        print("run:", i)
        clf_uniform = KNeighborsClassifier(n_neighbors = int(n), weights='uniform')
        clf_uniform.fit(X,y)
        cv_uniform_scores = cross_validate(clf_uniform,X,y, cv=cv)
        
        clf_weighted = KNeighborsClassifier(n_neighbors = int(n), weights = 'distance')
        clf_weighted.fit(X,y)
        cv_weighted_scores = cross_validate(clf_weighted,X,y, cv=cv)
    
        uniform_scores[i] = cv_uniform_scores.get('test_score')
        weighted_scores[i] = cv_weighted_scores.get('test_score')
        i += 1

    print("done with iterations")
    uniform_scores_mean = np.mean(uniform_scores, axis=1)
    uniform_scores_std = np.std(uniform_scores, axis=1)
    weighted_scores_mean = np.mean(weighted_scores, axis=1)
    weighted_scores_std = np.std(weighted_scores, axis=1)
    plt.grid()
    plt.plot(neighbors, uniform_scores_mean, '--', color="r",
             label="Unweighted score")
    plt.plot(neighbors, weighted_scores_mean, '--', color="g",
             label="Weighted score")

    plt.legend(loc="best")
    return plt

def plot_compare_svm(title,X, y, ylim=(0,1), cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    # estimators
    linear = SVC(kernel = 'linear', class_weight = 'balanced')
    poly2 = SVC(kernel = 'poly', degree = 2, class_weight = 'balanced')
    poly3 = SVC(kernel = 'poly', degree = 3, class_weight = 'balanced')
    rbf1 = SVC(kernel = 'rbf', gamma = 0.01, class_weight = 'balanced')
    rbf2 = SVC(kernel = 'rbf', gamma = 0.1, class_weight = 'balanced')
    rbf3 = SVC(kernel = 'rbf', gamma = 1.0, class_weight = 'balanced')
    
    colors = ['b','g','r','m','k','c']
    labels = ['linear','2nd polynomial', '3rd polynomial',' rbf gamma = 0.01','rbf gamma = 0.1', 'rbg = 1.0']
    estimators = [poly2, poly3, rbf1, rbf2, rbf3]
    i = 0
    for est in estimators:
        print('estimator : ', i)
        train_sizes, _, test_scores = learning_curve(est, X, y, cv=cv, 
                                                                n_jobs=n_jobs, train_sizes=train_sizes)
        test_scores_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, test_scores_mean, '--', color = colors[i], label=labels[i])
        i += 1
    
    plt.grid()
    plt.legend(loc="best")
    return plt

def plot_various_estimators(estimator, title, X, y, ylim=(0,1), cv=3, step_size= 10, min_estimators=1, max_estimators= 50):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of Estimators")
    plt.ylabel("Score")
                          
    num_estimators = np.linspace(min_estimators,max_estimators,step_size)
    train_scores = np.zeros([step_size,cv])
    test_scores = np.zeros([step_size,cv])
    i = 0
    
    for n in num_estimators:
        print("estimator:", i)
        booster = AdaBoostClassifier(estimator, n_estimators = int(n))
        booster.fit(X,y)
        cv_scores = cross_validate(booster,X,y, cv=cv)
        train_scores[i] = cv_scores.get('train_score')
        test_scores[i] = cv_scores.get('test_score')
        i += 1

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(num_estimators, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(num_estimators, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(num_estimators, train_scores_mean, '--', color="r",
             label="Training score")
    plt.plot(num_estimators, test_scores_mean, '--', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_nnet_compare_iters(estimators,title, X, Y, min_iters = 100, max_iters = 500, step_size= 5, cv=3):
    plt.figure()
    plt.grid()
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)

    iterations = np.linspace(min_iters, max_iters, step_size)
    for est in estimators:
        avg_scores = np.zeros([step_size])
        i=0
        for its in iterations:
            est.set_params(max_iter = int(its))
            scores = cross_val_score(est, X, Y, cv=cv)
            avg_scores[i] = np.mean(scores)
            i += 1
        label = 'Activation {0}, momentum {1}'.format(est.activation, est.momentum)    
        plt.plot(iterations, avg_scores, '--', label = label)
    plt.legend(loc ='best')
    
