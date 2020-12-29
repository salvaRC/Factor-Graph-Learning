import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, f1_score, recall_score, precision_score, \
    mean_squared_error, mean_absolute_error
from collections import Counter
from examples.utils import change_labels, to_int_label_array


def probs_to_preds(probs, eps=0.0, abstention=0, class_balance=None, neg_label=-1, pos_label=1):
    """
    eps - minimun distance from 0.5, defines confidence intervals
    """
    if class_balance is None:
        class_balance = [0.5, 0.5]
    predictions = probs.copy()
    if eps == 0.0:
        abstentions_idxs = predictions == 0.5
    else:
        uncertainty_interval = np.logical_and(0.5 + eps > predictions, predictions > 0.5 - eps)
    predictions[predictions >= 0.5 + eps] = pos_label
    predictions[predictions < 0.5 - eps] = neg_label
    if eps == 0.0:
        predictions[abstentions_idxs] = np.random.choice([neg_label, pos_label], np.count_nonzero(abstentions_idxs),
                                                         p=class_balance)
    else:
        predictions[uncertainty_interval] = abstention

    return predictions


def majority_voter_probs(L, n_classes=2, abst=0):
    n, m = L.shape
    Y_p = np.zeros((n, n_classes))
    for i in range(n):
        counts = np.zeros(n_classes)
        for j in range(m):
            if L[i, j] != abst:
                counts[L[i, j]] += 1
        Y_p[i, :] = np.where(counts == max(counts), 1, 0)
    Y_p /= Y_p.sum(axis=1).reshape(-1, 1)
    return Y_p


def eval_majority_vote(label_matrix, Y, MV_policy='random', abst=0, verbose=True, only_on_labeled=False):
    L = label_matrix if isinstance(label_matrix, np.ndarray) else label_matrix.toarray()
    L = L.copy()
    preds_MMVV = get_majority_vote(L, probs=True, abstention_policy=MV_policy, abstention=abst)
    # preds_MMV = majority_voter_probs(L, abst=abst)
    preds_MV = get_majority_vote(L, probs=False, abstention_policy=MV_policy, abstention=abst)
    print('Majority vote stats:')
    # stats = eval_final_predictions(Y, preds_MV, probs=preds_MMV[:, 1], abstention=abst, verbose=verbose,
    #                                only_on_labeled=only_on_labeled)
    stats = eval_final_predictions(Y, preds_MV, probs=preds_MMVV[:, 1], abstention=abst, verbose=verbose,
                                   only_on_labeled=only_on_labeled)
    return stats


def eval_final_predictions(Y, preds, probs=None, abstention=-1, only_on_labeled=True,
                           verbose=True, neg_label=-1, add_prefix="", parent_stats=None):
    predsC = preds.copy()
    Yc = Y.copy()
    probsC = predsC if probs is None else probs.copy()

    labeled = (preds != abstention)
    if only_on_labeled:
        predsC = preds[labeled]
        Yc = Y[labeled]
        probsC = probs[labeled]

    acc = accuracy_score(Yc, predsC)
    t = preds == Y
    tp = np.count_nonzero(np.logical_and(t, Y == 1))
    tn = np.count_nonzero(np.logical_and(t, Y == neg_label))
    fp = np.count_nonzero(np.logical_and(np.logical_not(t), preds == 1))
    fn = np.count_nonzero(np.logical_and(np.logical_not(t), preds == neg_label))
    recall = recall_score(Yc, predsC)
    precision = precision_score(Yc, predsC)
    f1 = f1_score(Yc, predsC)
    if abstention == 0:
        if verbose:
            print("Changing -1 <-> 0 for computing probability scores")
        absts = Yc == 0
        Yc[Yc == -1] = 0
        Yc[absts] = -1
    auc = roc_auc_score(Yc, probsC)
    logloss = log_loss(Yc, probsC)
    brier = np.square(Yc - probsC).mean()
    coverage = np.count_nonzero(labeled) / Y.shape[0]
    MSE = mean_squared_error(Yc, probsC)
    MAE = mean_absolute_error(Yc, probsC)
    stats = {'accuracy': acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn, 'recall': recall, 'precision': precision, 'f1': f1,
             "MSE": MSE, "MAE": MAE, 'auc': auc, 'logloss': logloss, 'brier': brier, 'coverage': coverage}
    if parent_stats is None:
        stats = {f"{add_prefix}_{key}": value for key, value in stats.items()}
    else:
        for key, value in stats.items():
            parent_stats[f"{add_prefix}_{key}"] = value
    if verbose:
        print('Accuracy:{:.3f} | Precision:{:.3f} | Recall:{:.3f} | F1 score:{:.3f} | AUC:{:.3f} | Log loss:{:.3f}'
              ' | Brier:{:.3f} | Coverage:{:.3f} | MSE, MAE:{:.3f}, {:.3f}'
              .format(acc, precision, recall, f1, auc, logloss, brier, coverage, MSE, MAE))
    if parent_stats is None:
        return stats
    else:
        return parent_stats


def samples_label_counts(label_matrix, upto=None, verbose=True):
    L = label_matrix.copy()
    L[L == -1] = 1
    counts = np.sum(L, axis=1)
    counts = Counter(counts)
    for key in counts:
        counts[key] /= L.shape[0]
    counts = counts.most_common(upto)

    if verbose:
        w = ', '.join(['{} votes:{:.3f}'.format(i, count) for i, count in counts])
        print('Fraction of samples with', w)
    return counts


def pred_and_eval_gen_model(model, label_matrix, y_true, verbose=True, print_MV=True, eps=1e-3, abst=0,
                            class_balance=None, counts_upto=5, version=7, return_preds=False, coverage_stats=True,
                            neg_label=None, pos_label=1, add_prefix="", parent_stats=None):
    mv_policy = 'random' if eps == 0.0 else 'abstain'
    only_on_labeled = eps != 0.0
    if isinstance(label_matrix, np.ndarray):
        if version in [10, 99]:
            prob_preds = model.predict_proba(label_matrix)
            if version == 10:
                prob_preds = prob_preds[:, 1]
        else:
            prob_preds = model.marginals(sparse.csr_matrix(label_matrix))
    else:
        prob_preds = model.marginals(label_matrix)
    if neg_label is None:
        neg_label = 0 if abst == -1 else -1
    preds_discrete = probs_to_preds(prob_preds, eps=eps, class_balance=class_balance, neg_label=neg_label,
                                    abstention=abst, pos_label=pos_label)

    stats1 = eval_final_predictions(y_true, preds_discrete, probs=prob_preds, abstention=abst, verbose=verbose,
                                    neg_label=neg_label, only_on_labeled=only_on_labeled, add_prefix=add_prefix, parent_stats=parent_stats)
    if coverage_stats:
        count_fracs = samples_label_counts(label_matrix, verbose=verbose, upto=counts_upto)
        for k, count_frac in count_fracs:
            stats1[f'With {k} votes'] = count_frac
    if print_MV and verbose:
        stats2 = eval_majority_vote(label_matrix, y_true, abst=abst, verbose=verbose, only_on_labeled=only_on_labeled)
        print("Gains over majority vote:\nAccuracy:{:.4f}, Recall: {:.4f}, Precision: {:.4f}, F1 score: {:.4f}"
              ", AUC: {:.4f}".format((stats1['accuracy'] - stats2['accuracy']), (stats1['recall'] - stats2['recall']),
                                     (stats1['precision'] - stats2['precision']), (stats1['f1'] - stats2['f1']),
                                     (stats1['auc'] - stats2['auc'])))
        if return_preds:
            return stats1, stats2, preds_discrete
        return stats1, stats2
    if return_preds:
        return stats1, prob_preds
    return stats1


def get_majority_vote(label_matrix, probs=False, n_classes=2, abstention_policy='drop', abstention=-1, metal=False):
    def majority_vote(row, abst=0):
        tmp = np.zeros(n_classes)
        if metal:
            for i in row:
                tmp[i - 1] += 1 if i != abst else 0
        else:
            for i in row:
                tmp[i] += 1 if i != abst else 0

        if not tmp.any():
            if probs:
                return np.ones(n_classes) / n_classes  # return uniform probs
            else:
                return abstention
        elif probs:
            res = np.zeros(n_classes)
            res[np.argmax(tmp)] = 1
            return res
            return (tmp / len(row)).reshape(1, -1)
        else:
            pred = np.argmax(tmp)
            if abstention == 0 and pred == 0:
                pred = -1
            return pred

    '''
    Equivalent to:
    maj_voter = MajorityLabelVoter()
    majority_preds = maj_voter.predict_proba(label_matrix)
    majority_preds = np.argmax(majority_preds[idxs], axis=1)
    accMV, precMV, recMV, f1MV = iws.utils.eval_final_predictions(majority_preds, Ytrain)
    '''
    label_mat = label_matrix.copy()
    if abstention == 0:  # better map 0 -> -1 and vice versa probably
        label_mat = change_labels(label_mat, old_label=0, new_label=-1)
        votes = np.apply_along_axis(majority_vote, 1, label_mat, -1)
    else:
        votes = np.apply_along_axis(majority_vote, 1, label_mat, abstention)
    if probs:
        votes = votes.reshape(votes.shape[0], n_classes)

    if abstention_policy == 'drop':
        votes = votes[votes != abstention]
    elif abstention_policy == 'random' and not probs:
        votes[votes == abstention] = np.random.choice([-1, 1], np.count_nonzero(votes == abstention))
    return votes


def lf_empirical_accuracies(L, Y: np.ndarray, abst_label=0) -> np.ndarray:
    """
                    Taken from Snorkel v0.9
    Compute empirical accuracy against a set of labels Y for each LF.

            Usually, Y represents development set labels.

            Parameters
            ----------
            Y
                [n] or [n, 1] np.ndarray of gold labels

            Returns
            -------
            numpy.ndarray
                Empirical accuracies for each LF
    """
    L = L.toarray() if not isinstance(L, np.ndarray) else L
    if abst_label == 0:
        L, Y = change_labels(L, Y, new_label=-1, old_label=0)
    Y = to_int_label_array(Y)
    X = np.where(
        L == -1,
        0,
        np.where(L == np.vstack([Y] * L.shape[1]).T, 1, -1),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(0.5 * (X.sum(axis=0) / (L != -1).sum(axis=0) + 1))


def lf_coverages(L, abstention=0):
    return np.ravel((L != abstention).sum(axis=0)) / L.shape[0]
