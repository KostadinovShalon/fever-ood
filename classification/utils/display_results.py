import numpy as np
import sklearn.metrics as sk

recall_level_default = 0.95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """
    Use high precision for cumulative sum and check that final value matches sum.

    Parameters
    ----------
    arr : array-like
        Array to be cumulatively summed as flat.
    rtol : float, optional
        Relative tolerance, see ``np.allclose``.
    atol : float, optional
        Absolute tolerance, see ``np.allclose``.

    Returns
    -------
    out : ndarray
        Cumulative sum of the input array.

    Raises
    ------
    RuntimeError
        If the cumulative sum is found to be unstable.
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    """
    Calculate the false positive rate (FPR) and false discovery rate (FDR) at a given recall level.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Target scores, can either be probability estimates of the positive class or confidence values.
    recall_level : float, optional
        The recall level at which to calculate FPR and FDR.
    pos_label : int or float, optional
        The label of the positive class. If None, the positive class is inferred from `y_true`.

    Returns
    -------
    fpr : float
        False positive rate at the given recall level.
    """
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    """
    Calculate AUROC, AUPR, and FPR at a given recall level.

    Parameters
    ----------
    _pos : array-like
        Scores for the positive class.
    _neg : array-like
        Scores for the negative class.
    recall_level : float, optional
        The recall level at which to calculate FPR.

    Returns
    -------
    auroc : float
        Area under the receiver operating characteristic curve.
    aupr : float
        Average precision score.
    fpr : float
        False positive rate at the given recall level.
    """
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def show_performance(pos, neg, method_name='Ours', recall_level=recall_level_default):
    """
    Display performance metrics including AUROC, AUPR, and FPR.

    Parameters
    ----------
    pos : array-like
        Scores for the positive class.
    neg : array-like
        Scores for the negative class.
    method_name : str, optional
        Name of the method being evaluated.
    recall_level : float, optional
        The recall level at which to calculate FPR.
    """
    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))


def print_measures(auroc, aupr, fpr, method_name='Ours', recall_level=recall_level_default):
    """
    Print performance metrics including AUROC, AUPR, and FPR.

    Parameters
    ----------
    auroc : float
        Area under the receiver operating characteristic curve.
    aupr : float
        Average precision score.
    fpr : float
        False positive rate at the given recall level.
    method_name : str, optional
        Name of the method being evaluated.
    recall_level : float, optional
        The recall level at which to calculate FPR.
    """
    print('\t\t\t\t' + method_name)
    print('  FPR{:d} AUROC AUPR'.format(int(100 * recall_level)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100 * fpr, 100 * auroc, 100 * aupr))


def print_measures_with_std(aurocs, auprs, fprs, method_name='Ours', recall_level=recall_level_default):
    """
    Print performance metrics including AUROC, AUPR, and FPR with standard deviations.

    Parameters
    ----------
    aurocs : array-like
        List of AUROC scores.
    auprs : array-like
        List of AUPR scores.
    fprs : array-like
        List of FPR scores.
    method_name : str, optional
        Name of the method being evaluated.
    recall_level : float, optional
        The recall level at which to calculate FPR.
    """
    print('\t\t\t\t' + method_name)
    print('  FPR{:d} AUROC AUPR'.format(int(100 * recall_level)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100 * np.mean(fprs), 100 * np.mean(aurocs), 100 * np.mean(auprs)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100 * np.std(fprs), 100 * np.std(aurocs), 100 * np.std(auprs)))


def show_performance_comparison(pos_base, neg_base, pos_ours, neg_ours, baseline_name='Baseline',
                                method_name='Ours', recall_level=recall_level_default):
    """
    Display performance comparison between baseline and current method.

    Parameters
    ----------
    pos_base : array-like
        Scores for the positive class from the baseline method.
    neg_base : array-like
        Scores for the negative class from the baseline method.
    pos_ours : array-like
        Scores for the positive class from the current method.
    neg_ours : array-like
        Scores for the negative class from the current method.
    baseline_name : str, optional
        Name of the baseline method.
    method_name : str, optional
        Name of the current method.
    recall_level : float, optional
        The recall level at which to calculate FPR.
    """
    auroc_base, aupr_base, fpr_base = get_measures(pos_base[:], neg_base[:], recall_level)
    auroc_ours, aupr_ours, fpr_ours = get_measures(pos_ours[:], neg_ours[:], recall_level)

    print('\t\t\t' + baseline_name + '\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}\t\t{:.2f}'.format(
        int(100 * recall_level), 100 * fpr_base, 100 * fpr_ours))
    print('AUROC:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * auroc_base, 100 * auroc_ours))
    print('AUPR:\t\t\t{:.2f}\t\t{:.2f}'.format(
        100 * aupr_base, 100 * aupr_ours))
