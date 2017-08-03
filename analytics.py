import pandas as pd
import numpy as np
from scipy import stats
from sklearn import model_selection, tree
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings

### List of metrics analysed in the paper  ###
# The 'combined' list has all the 22 metrics
feature_names_combined = (
    'entities', 'agents', 'activities',  # PROV types (for nodes)
    'nodes', 'edges', 'diameter', 'assortativity',  # standard metrics
    'acc', 'acc_e', 'acc_a', 'acc_ag',  # average clustering coefficients
    'mfd_e_e', 'mfd_e_a', 'mfd_e_ag',  # MFDs
    'mfd_a_e', 'mfd_a_a', 'mfd_a_ag',
    'mfd_ag_e', 'mfd_ag_a', 'mfd_ag_ag',
    'mfd_der',  # MFD derivations
    'powerlaw_alpha'  # Power Law
)
# The 'generic' list has 6 generic network metrics (that do not take provenance information into account)
feature_names_generic = (
    'nodes', 'edges', 'diameter', 'assortativity',  # standard metrics
    'acc',
    'powerlaw_alpha'  # Power Law
)
# The 'provenance' list has 16 provenance-specific network metrics
feature_names_provenance = (
    'entities', 'agents', 'activities',  # PROV types (for nodes)
    'acc_e', 'acc_a', 'acc_ag',  # average clustering coefficients
    'mfd_e_e', 'mfd_e_a', 'mfd_e_ag',  # MFDs
    'mfd_a_e', 'mfd_a_a', 'mfd_a_ag',
    'mfd_ag_e', 'mfd_ag_a', 'mfd_ag_ag',
    'mfd_der',  # MFD derivations
)
# The utitility of above threes set of metrics will be assessed in our experiements to
# understand whether provenance type information help us improve data classification performance
feature_name_lists = (
    ('combined', feature_names_combined),
    ('generic', feature_names_generic),
    ('provenance', feature_names_provenance)
)


def balance_smote(df):
    X = df.drop('label', axis=1)
    Y = df.label
    print('Original data shapes:', X.shape, Y.shape)
    
    smoX, smoY = X, Y
    c = Counter(smoY)
    while (min(c.values()) < max(c.values())):  # check if all classes are balanced, if not balance the first minority class
        smote = SMOTE(ratio="auto", kind='regular')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            smoX, smoY = smote.fit_sample(smoX, smoY)
        c = Counter(smoY)
    
    print('Balanced data shapes:', smoX.shape, smoY.shape)
    df_balanced = pd.DataFrame(smoX, columns=X.columns)
    df_balanced['label'] = smoY
    return df_balanced


def t_confidence_interval(an_array, alpha=0.95):
    s = np.std(an_array)
    n = len(an_array)
    return stats.t.interval(alpha=alpha, df=(n - 1), scale=(s / np.sqrt(n)))


def cv_test(X, Y, n_iterations=1000, test_id=""):
    accuracies = []
    importances = []
    while len(accuracies) < n_iterations:
        skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True)
        for train, test in skf.split(X, Y):
            clf = tree.DecisionTreeClassifier()
            clf.fit(X.iloc[train], Y.iloc[train])
            accuracies.append(clf.score(X.iloc[test], Y.iloc[test]))
            importances.append(clf.feature_importances_)
    print("Accuracy: %.2f%% ±%.4f <-- %s" % (np.mean(accuracies) * 100, t_confidence_interval(accuracies)[1] * 100, test_id))
    return accuracies, importances


def test_classification(df, n_iterations=1000, test_id=''):
    results = pd.DataFrame()
    imps = pd.DataFrame()
    Y = df.label
    for feature_list_name, feature_names in feature_name_lists:
        X = df[list(feature_names)]
        accuracies, importances = cv_test(
            X, Y, n_iterations, '-'.join((test_id, feature_list_name)) if test_id else feature_list_name
        )
        rs = pd.DataFrame(
            {
                'Metrics': feature_list_name,
                'Accuracy': accuracies
            }
        )
        results = results.append(rs, ignore_index=True)
        if feature_list_name == "combined":  # we are interested in the relevance of all features (i.e. 'combined') 
            imps = pd.DataFrame(importances, columns=feature_names)
    return results, imps