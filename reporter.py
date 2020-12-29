from collections import OrderedDict
import numpy as numpy
import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt 

from sklearn import metrics, calibration
from sklearn.preprocessing import binarize, normalize

from IPython.display import Markdown, display
import numpy as np

def calibration_curve(expected, proba_predicted, n_bins=10):
    fpv, mpv = calibration.calibration_curve(expected, proba_predicted,
                                            n_bins=n_bins)
    return pd.DataFrame({'fpv': fpv, 'mpv': mpv})

def roc_curve(expected, proba_predicted, pos_label=None):

    fpr, tpr, threshold = metrics.roc_curve(expected, proba_predicted,
                                            pos_label=pos_label)

    curve = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold,
                         'distance': np.sqrt(fpr ** 2 + (tpr - 1) ** 2)})

    return curve

def precision_recall_curve(expected, proba_predicted, pos_label=None):

    precision, recall, threshold = metrics.precision_recall_curve(
        expected, proba_predicted, pos_label=pos_label)

    threshold = np.append(threshold, threshold[-1])
    curve = pd.DataFrame({'precision': precision, 'recall': recall,
                         'threshold': threshold})

    return curve.sort_values(by='recall')

def best_threshold(expected, proba_predicted, pos_label=None):

    fpr, tpr, threshold = metrics.roc_curve(expected, proba_predicted,
                                            pos_label=pos_label)

    curve = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold,
                         'distance': np.sqrt(fpr ** 2 + (tpr - 1) ** 2)})
    
    #return curve['threshold'][np.argmin(curve['distance'])]
    return curve['threshold'][curve['distance'].idxmin()]

def classification_report_table(expected, predicted, digits=4, set_name=None):


    str_rep = metrics.classification_report(expected, predicted, digits=digits)
    splitted = str_rep.split('\n')

    result = list()
    for line in splitted[2:4]:
        cols = line.split()
        
        row = OrderedDict(Set=set_name, Label=cols[0], Precision=cols[1],
                          Recall=cols[2], F1=cols[3], Support=cols[4])

        result.append(pd.DataFrame(row, index=[0]))
    
    
    cols = ['Avg/Total'] + splitted[-2].split()[2:]  
    row = OrderedDict(Set=set_name, Label=cols[0], Precision=cols[1],
                    Recall=cols[2], F1=cols[3], Support=cols[4])

    df_row = pd.DataFrame(row, index=[0])
    return pd.concat(result + [df_row])


def ks_test(expected, proba_predicted, pos_label=None, threshold_samples=100):   

    if pos_label is None:
        pos_label = 1

    _, _, threshold = metrics.roc_curve(expected, proba_predicted,
                                        pos_label=pos_label)

    curve = pd.DataFrame({'expected': expected,
                         'proba_predicted': proba_predicted})

    curve.sort_values(by='proba_predicted')
    cumulative = list()
    total_good = np.sum(curve['expected'] != pos_label)
    total_bad = np.sum(curve['expected'] == pos_label)

    min_, max_ = np.min(proba_predicted), np.max(proba_predicted)
    threshold = threshold[(threshold >= min_) & (threshold <= max_)]
    indices = np.floor(np.arange(0, 1, 0.01) * len(threshold)).astype(np.int64)

    for thresh in threshold[indices]:
        bucket = np.where(curve['proba_predicted'] <= thresh)
        labels = curve['expected'].iloc[bucket]
        cumulative.append({'good': np.sum(labels != pos_label) / total_good,
                            'bad': np.sum(labels == pos_label) / total_bad,
                            'threshold': thresh})

    return pd.DataFrame(cumulative).sort_values(by='threshold')


class Reporter(object):

    
    PALETTE = [
        ['maroon', 'firebrick', 'red', 'orangered', 'coral', 'chocolate'],
        ['darkgreen', 'forestgreen', 'green', 'limegreen', 'mediumseagreen',
        'lime'],
        ['midnightblue', 'darkblue', 'blue', 'dodgerblue', 'cyan', 'blue'],
        ['darkgoldenrod', 'darkorange', 'orange', 'gold', 'gold', 'yellow'],
        ['rebeccapurple', 'darkviolet', 'purple', 'blueviolet', 'violet', 'magenta']]

    '''REPORTS = ['summary', 'scores', 'confusion_matrix', 'roc_curve',
              'precision_recall', 'ks_test', 'calibration_curve']'''

    REPORTS = ['summary', 'scores', 'confusion_matrix', 'roc_curve',
            'precision_recall', 'ks_test', 'calibration_curve']          


    TITLE_FORMAT = '### {}'  

    def __init__(self, plot_scale=1, dpi=90, digits=4, **kwargs):

        self.results = dict()
        self.plot_scale = plot_scale
        self.dpi = dpi
        self.digits = digits

    def _title(self, title):
        display(Markdown(self.TITLE_FORMAT.format(title)))

    def put(self, name, expected, proba_predicted, threshold=None,
            pos_label=None, palette=None):
        
        if threshold is None:
           threshold = best_threshold(expected, proba_predicted)

        row_proba_predicted = proba_predicted.reshape(1, -1)
        predicted = binarize(row_proba_predicted, threshold).reshape(-1, 1)
        self.results[name] = {'expected': expected,
                              'proba_predicted': proba_predicted,
                              'predicted': predicted, 'threshold': threshold,
                              'pos_label': pos_label, 'palette': palette}
                              

    def summary(self):

        summ = list()
        self._title('Summary')
        for name, result in self.results.items():
            summ.append(classification_report_table(
                result['expected'], result['predicted'],
                digits=self.digits, set_name=name))

        summary = pd.concat(summ).set_index(['Set', 'Label'], drop=True)
        display(summary)
        return summary

    def scores(self, **kwargs):

        def is_enabled(name, default=True):
            if ('all' in kwargs and kwargs['all']) or \
               ('all_' in kwargs and kwargs['all_']):
               return True

            if not kwargs and default:
                return True
            return (name in kwargs) and kwargs[name]    

        scores= list()
        self._title('Scores')
        for name, result in self.results.items():
            exp, pred = result['expected'], result['predicted']
            proba = result['proba_predicted']
            row = list()
            if is_enabled('accuracy'):
                row.append(('Accuracy', metrics.accuracy_score(exp, pred)))

            if is_enabled('kappa'):
                row.append(('Cohen K', metrics.cohen_kappa_score(exp, pred)))

            if is_enabled('auc'):
                fpr, tpr, _ = metrics.roc_curve(exp, proba,
                                                pos_label=result['pos_label'])
                row.append(('AUCROC', metrics.auc(fpr,tpr)))

            if is_enabled('auc_micro', default=False):
               auc = metrics.roc_auc_score(exp, proba, average='micro')
               row.append(('Micro AUCROC', auc))

            if is_enabled('auc_macro', default=False):
               auc = metrics.roc_auc_score(exp, proba, average='macro')
               row.append(('Macro AUCROC', auc))

            if is_enabled('auc_weighted', default=False):
               auc = metrics.roc_auc_score(exp, proba, average='weighted')
               row.append(('Weighted AUCROC', auc))

            if is_enabled('brier'):
                score = metrics.brier_score_loss(exp, proba,
                                                pos_label=result['pos_label'])

                row.append(('Brier loss', score))

            if is_enabled('matthews'):
                score = metrics.matthews_corrcoef(exp, pred)                                        
                row.append(('Matthews CC', score))

            if is_enabled('fbeta_precision', default=False):
                score = metrics.fbeta_score(exp, pred, beta=0.5)
                row.append(('F-(B=0.5 Precision', score))


            if is_enabled('fbeta_recall', default=False):
                score = metrics.fbeta_score(exp, pred, beta=10)
                row.append(('F-(B=10 Recall', score))

            if is_enabled('f1_binary', default=False):
                score = metrics.f1_score(exp, pred, average='binary')
                row.append(('F1-binary', score))

            if is_enabled('f1_micro', default=False):
                score = metrics.f1_score(exp, pred, average='micro')
                row.append(('F1-binary', score))

            if is_enabled('f1_macro', default=False):
                score = metrics.f1_score(exp, pred, average='macro')
                row.append(('F1-macro', score))


            if is_enabled('f1_weighted'):
                score = metrics.f1_score(exp, pred, average='weighted')
                row.append(('F1-weighted', score))

            scores.append(pd.DataFrame(OrderedDict(row), index=[name]))

        scores_table = pd.concat(scores)
        display(scores_table.round(self.digits))
        return scores_table   


    def confusion_matrix(self, normalized=True):

        rows = len(self.results)
        cols = 2 if normalized else 1
        sizes = ((2 * cols + cols), 2 * cols)
        fig, ax = plt.subplots(rows, cols, figsize=sizes, dpi=self.dpi)
        if not normalized and rows == 1:
            ax = [ax]

        fig.subplots_adjust(hspace=1)
        row = 0
        self._title('Confusion Matrix')
        for name, result in self.results.items():
            expected, predicted = result['expected'], result['predicted']
            vmax = np.max(np.bincount(expected))
            mat = metrics.confusion_matrix(expected, predicted)
            axis = ax[row][0] if normalized and rows > 1 else ax[row]
            sns.heatmap(mat, annot=True, vmax=vmax, fmt='d', cbar=False,
                        ax=axis)
            axis.set_xlabel('{} Predicted Label'.format(name))
            axis.set_ylabel('{} True Label'.format(name))
            axis.set_title('{} Matrix'.format(name))
            if normalized:
                norm_mat = normalize(mat.astype(np.float64), axis=1, norm='l1')
                axis = ax[row][1] if rows > 1 else ax[row + 1]
                sns.heatmap(norm_mat, vmin=0.0, vmax=1.0, annot=True,
                            fmt='.{}f'.format(self.digits), ax=axis)
                axis.set_xlabel('{} Predicted Label'.format(name))
                axis.set_ylabel('{} True Label'.format(name))
                axis.set_title('{} Normalized Matrix'.format(name))
            row += 1
        plt.tight_layout()
        plt.show()

    def roc_curve(self, over_threshold=False, **kwargs):

        width = 1
        if over_threshold:
            width = 2

        sizes = (5 * width * self.plot_scale, 5 * self.plot_scale)
        _, ax = plt.subplots(1, width, figsize=sizes, dpi=self.dpi)

        lw_main, lw_dist, lw_cut = 2, 1.5, 1
        if over_threshold:
            roc, thr = ax
        else:
            roc = ax

        self._title('ROC Curve Analysis')
        result_index = 0
        for name, result in self.results.items():
            if result['palette']:
                palette = result['palette']
            else:
                palette = self.PALETTE[result_index]

            #Scores
            curve = roc_curve(result['expected'], result['proba_predicted'])
            auc = metrics.auc(curve['fpr'], curve['tpr'])
            fmt = '- {{}} Gini coefficient: {{:.{}f}}'.format(self.digits)
            display(Markdown(fmt.format(name, 2.0 * (auc - 0.5))))
            #First plot
            roc.plot(curve['fpr'], curve['distance'], lw=lw_dist,
                      color=palette[4], linestyle=':',
                      label='{} distance to (0, 1)'.format(name))

            roc.plot(curve['fpr'], curve['tpr'], lw=lw_main, color=palette[2],
                    label='{} ROC Curve (area = {:.2f})'.format(name, auc))

            
            cut = np.abs(curve['threshold'] - result['threshold']).idxmin()
            
            roc.axvline(x=curve['fpr'][cut], lw=lw_cut, color=palette[5],
                        linestyle='-.',
                        label='{} threshold ({:.2f})'.format(
                            name,  result['threshold']))

            if result_index == 0:
                roc.plot([0,1], [0,1], color='navy', lw=lw_main,
                        linestyle='--')

            roc.set_xlim([0.0, 1.0])
            roc.set_ylim([0.0, 1.05])
            roc.set_xlabel('False Positive Rate')
            roc.set_ylabel('True Positive Rate')
            roc.set_title('{} Receiver Operating Characteristic'.format(name))
            roc.legend(loc='lower right')
            # Second plot
            if over_threshold:
                curve.sort_values(by='threshold', inplace=True)
                thr.plot(curve['threshold'], curve['distance'], lw=lw_dist,
                        color=palette[4], linestyle=':',
                        label='{} distance to (0, 1)'.format(name))
                thr.plot(curve['threshold'], curve['fpr'], lw=lw_main,
                        color=palette[4], linestyle='--',
                        label='{} FPR'.format(name))
                thr.plot(curve['threshold'], curve['tpr'], lw=lw_main,
                        color=palette[0], label='{} TPR'.format(name))

                thr.axvline(x=curve['threshold'][cut], lw=lw_cut, color=palette[5],
                linestyle='-.',
                label='{} threshold ({:.2f})'.format(
                    name, result['threshold']))

                thr.set_ylim([0.0, 1.0])
                thr.set_xlabel('Threshold')
                thr.set_ylabel('Ratio')
                thr.set_title(loc='center left', bbox_to_anchor=(1, 0.5))

            result_index += 1

        plt.tight_layout()
        plt.show()        

    def ks_test(self, **kwargs):

        self._title('Kolmogorov-Smirnov test')
        sizes = (4 * self.plot_scale, 4 * self.plot_scale)
        plt.figure(figsize=sizes, dpi=self.dpi)
        lw_main, lw_dist = 2, 1.5
        result_index = 0
        for name, result in self.results.items():
            if result['palette']:
                palette = result['palette']

            else:
                palette = self.PALETTE[result_index]

            plt.title('Kolmogorov-Smirnov test') 
            plt.ylabel('Cumulative Probability')
            plt.xlabel('X')
            curve = ks_test(result['expected'], result['proba_predicted'],
                            pos_label=result['pos_label'])

            plt.plot(curve['threshold'], curve['good'], lw=lw_main,
                    color=palette[0], linestyle='--',
                    label='{} Good Rate'.format(name))

            plt.plot(curve['threshold'], curve['bad'], lw=lw_main,
                    color=palette[3], linestyle='-',
                    label='{} Bad Rate'.format(name))

            plt.ylim([0.0, 1.05])
            dist = np.abs(curve['bad'] - curve['good'])
            max_dist = np.max(dist)
            cut = dist.idxmax()
            ymin = curve['bad'][cut]
            ymax = curve['good'][cut]
            if ymin > ymax:
                ymin = curve['good'][cut]
                ymax = curve['bad'][cut]

            thres_cut = curve['good'][cut]
            plt.axvline(x=thres_cut, lw=lw_dist, color=palette[5],
                        linestyle='-.', ymin=ymin + 0.03, ymax=ymax - 0.03,
                        label='{} K-S statistic ({:.2f})'.format(
                            name, max_dist))

            plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
            fmt = '- {{}} K-S statistic: {{:.{}f}}'.format(self.digits)
            display(Markdown(fmt.format(name, max_dist)))
            result_index += 1

        plt.tight_layout()
        plt.show()

    def precision_recall(self, over_threshold=False, **kwargs):  

        self._title('Precision-Recall Curve')
        width = 1
        if over_threshold:
            width = 2
        sizes = (5 * width * self.plot_scale, 5 * self.plot_scale)
        _, ax = plt.subplots(1, width, figsize=sizes, dpi=self.dpi)
        lw_main = 2
        result_index = 0
        for name, result in self.results.items():
            if result['palette']:
                palette = result['palette']
            else:
                palette = self.PALETTE[result_index]

            curr_ax = ax[0] if over_threshold else ax
            #First plot
            curr_ax.set_title('Precision-Recall')
            curr_ax.set_ylabel('Precision')
            curr_ax.set_xlabel('Recall')
            curve = precision_recall_curve(
                result['expected'], result['proba_predicted'],
                pos_label=result['pos_label'])

            by_recall = curve.copy().sort_values(by='recall')
            auc = metrics.auc(by_recall['recall'], by_recall['precision'])
            curr_ax.plot(curve['recall'], curve['precision'], lw=lw_main,
                        color=palette[1],
                        label='{} Precision=Recall curve (area = {:.2f})'.format(
                            name, auc))

            curr_ax.legend(loc='lower left')
            if over_threshold:
                ax[1].set_title('Precision & Recall over Threshold')
                ax[1].set_ylabel('Precision & Recall')
                ax[1].set_xlabel('Threshold')
                by_thres = curve.copy().sort_values(by='threshold')
                ax[1].plot(by_thresh['threshold'], by_thresh['precision'], lw=lw_main,
                           color=palette[3], linestyle='--',
                           label='{} Precision'.format(name))
                ax[1].plot(by_thresh['threshold'], by_thresh['recall'], lw=lw_main,
                          color=palette[1],
                          label='{} Recall'.format(name))

                ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            result_index += 1

        plt.tight_layout()
        plt.show()                             

    def calibration_curve(self, n_bins=10, **kwargs):

        self._title('Probability Calibration - Reliability Curve')
        fig = plt.figure(figsize=(6 * self.plot_scale, 6 * self.plot_scale), dpi=self.dpi)
        ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
        ax2 = plt.subplot2grid((3,1), (2,0))
        ax1.plot([0,1], [0,1], "k:", label="Perfectly calibrated")
        for name, result in self.results.items():
            expected = result['expected']
            predicted = result['predicted']
            proba_predicted = result['proba_predicted']

            curve = calibration_curve(expected, proba_predicted, n_bins=n_bins)

            clf_score = metrics.brier_score_loss(expected, proba_predicted,
                                                    pos_label=result['pos_label'])

            ax1.plot(curve['mpv'], curve['fpv'], "s-", markersize=3,
                    label="{} (Brier-{:4f})".format(name, clf_score)) 

            ax2.hist(proba_predicted, range=(0,1), bins=n_bins, label=name,
                    histtype="step", lw=2)


        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Probability Calibration - Reliability Curve')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()

    def report(self, **kwargs):
        def is_enabled(name):
            return (not kwargs) or (name in kwargs) and kwargs[name]

        if not self.results:
           print('No results were added to this Reporter.')
           print('Use reporter.put("Train", y_train,y_score) before.')
           return
        setup = dict()
        if 'setup' in kwargs:
            setup = kwargs['setup']
            del kwargs['setup']
        for name in self.REPORTS:
            if is_enabled(name):
                reporter = getattr(self, name)
                if name in setup:
                    if setup[name]:
                        reporter(**setup[name])
                else:
                    reporter()  


def report(proba_predictor, *args, **kwargs):

    reporter = Reporter(**kwargs)
    threshold = None
    for name, X, y in args:
        if not threshold:
            threshold = best_threshold(y, proba_predicted(X))
        reporter.put(name, y, proba_predictor(X), threshold=threshold)

    reporter.report(**kwargs)

def report_binary(predictor, *args, **kwargs):

    reporter = Reporter(**kwargs)
    threshold = None
    for name, X, y in args:
        reporter.put(name, y, predictor(X), threshold=0.5)

    reporter.report(**kwargs)






















