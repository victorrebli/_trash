import numpy as np
import pandas as pd
from sklearn import metrics, calibration
from sklearn.preprocessing import binarize, normalize
import matplotlib.pyplot as plt
import itertools


def best_threshold(expected, proba_predicted, pos_label=None):

    fpr, tpr, threshold = metrics.roc_curve(expected, proba_predicted,
                                            pos_label=pos_label)

    curve = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold,
                         'distance': np.sqrt(fpr ** 2 + (tpr - 1) ** 2)})

    return curve['threshold'][np.argmin(curve['distance'])]



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
    
    
def compute_roc(y_true, y_pred, plot=False):

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc_score = metrics.auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score


def feature_importance(clf, COLS):
    
    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(7, 6))
    plt.barh(pos[-10:], feature_importance[sorted_idx][-10:], align='center')
    plt.yticks(pos[-10:], COLS[sorted_idx][-10:])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show() 


def find_leave(clf, X_test, sample_id, verbose = False):

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_indicator = clf.decision_path(X_test)
    #print(node_indicator)
    leave_id = clf.apply(X_test)
    sample_id = sample_id
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                       node_indicator.indptr[sample_id + 1]]

    
    if verbose:
        print('Rules used to predict sample %s'  % sample_id)

    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            if verbose:
                print("leaf node {} reached, no decision here".format(leave_id[sample_id]))
            return leave_id[sample_id]
        else:
            if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            if verbose:
               print("decision id node %s : (X[%s, %s] (= %s) %s %s"
                    % (node_id,
                    sample_id,
                    feature[node_id],
                    X_test[sample_id, feature[node_id]],
                    threshold_sign,
                    threshold[node_id]))                                                        


def find_rules(clf, X_test, COLS):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    def find_path(node_numb, path, x):
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False
        if (children_left[node_numb] != -1):
            left = find_path(children_left[node_numb], path, x)
        if (children_right[node_numb] != -1):
            right = find_path(children_right[node_numb], path, x)
        if left or right:
            return True
        path.remove(node_numb)
        return False

    def get_rule(path, column_names):
        mask = ''
        for index, node in enumerate(path):
            if index != len(path) -1:
                if (children_left[node] == path[index+1]):
                    mask += "(X_test['{}']<= {}) \t ".format(column_names[feature[node]], 
                            threshold[node])
                else:
                    mask += "(X_test['{}']> {}) \t".format(column_names[feature[node]],
                                threshold[node])

        mask = mask.replace("\t", "&", mask.count("\t") - 1)
        mask = mask.replace("\t", "")
        return mask                
    
    
    leave_id = clf.apply(X_test)
    
    paths = {}
    for leaf in np.unique(leave_id):
        path_leaf = []
        find_path(0, path_leaf, leaf)
        paths[leaf] = np.unique(np.sort(path_leaf))
        
    rules = {}
    for key in paths:
        rules[key] = get_rule(paths[key], COLS)
        
    return rules    


def create_bins(data, bins):
    
    pred_bins = pd.cut(
        data, bins, labels=list(map(lambda x: '{}'.format(x), bins))[1:], precision=0)
    return pred_bins





