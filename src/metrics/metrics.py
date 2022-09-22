from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


def compute_confidence(preds):
    return torch.max(preds, -1)[0]


def computeMeanConfidence(preds):
    bla = torch.max(preds, -1)[0]
    return torch.mean(bla, 0)


def normed_entropy(preds: torch.Tensor, num_classes: Union[int, torch.Tensor]):
    entropy = -torch.sum(preds * torch.log(preds), dim=-1) / torch.log(torch.tensor(num_classes))
    # entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min())
    return entropy


def computeNormEntropy(preds, num_classes):
    return -torch.sum(preds * torch.log(preds), -1) / torch.tensor([num_classes]).float().log()


def computeKL(preds1, preds2):
    return torch.sum(preds1 * torch.log(preds1) - preds1 * torch.log(preds2), 1)


def computeChiSquare(preds1, preds2):
    return (preds1 - preds2) ** 2 / (preds1 + preds2)


def computeACC(preds, labels):
    argm = torch.argmax(preds, 1)
    corr = torch.sum(argm == labels)

    return corr / float(preds.size(0))


def computeTopKACC(preds, labels, k=5):
    argm_k = torch.topk(preds, k, 1)[1]
    corr_k = torch.sum(argm_k == labels.unsqueeze(-1), -1)

    corr_k = torch.sum(corr_k)

    return corr_k / float(preds.size(0))


def computeJSDiv(preds, comb_pred):  # Jensen-Shannon Divergence (Deep Ensembles a Loss Landscape perspective)
    cum_js = 0.0
    for pred in preds:
        cum_js += computeKL(pred, comb_pred)

    cum_js /= preds.size(0)  # Divide by number of ensemble members
    return cum_js


def computeBrier(preds, label, num_classes):
    # Geht von aus das alles gesoftmaxed ist
    # Letzte dimension sollte Klassen sein

    onehot_label = F.one_hot(label, num_classes)

    return torch.sum((preds - onehot_label) ** 2)


def computeNLL(preds, label):
    return F.nll_loss(torch.log(preds), label, reduction="sum")


def computeOracleNLL(preds, label):
    num_models = preds.size(0)

    tmp_nll = torch.empty((num_models, *label.size()), dtype=preds.dtype)

    for i in range(num_models):
        tmp_nll[i] = F.nll_loss(torch.log(preds[i]), label, reduction="none")

    min_nll = tmp_nll.min(dim=0)[0]

    return torch.sum(min_nll)


def computeConfAcc(preds, label):
    corr_ind = torch.argmax(preds, dim=1) == label
    cali = torch.mean(torch.abs(torch.max(preds, 1)[0] - corr_ind.float()))
    return cali


def calibrationBins(preds, uncert, labels, num_bins, score_range=[0, 1]):
    arg_preds = torch.argmax(preds, 1)

    max = score_range[1]
    min = score_range[0]

    width = (max - min) / num_bins

    bins = np.linspace(min, max, num_bins + 1)

    # bins = [{0: b * width + min, 1: (b + 1) * width + min} for b in range(num_bins)]
    # print(bins)

    accs = torch.zeros(num_bins, dtype=torch.int)
    nums = torch.zeros(num_bins, dtype=torch.int)
    avg_uncer = torch.zeros(num_bins, dtype=torch.float32)

    for i in range(num_bins):
        accs[i] = (
            labels[(bins[i] <= uncert) & (uncert < bins[i + 1])]
            == arg_preds[(bins[i] <= uncert) & (uncert < bins[i + 1])]
        ).sum()
        nums[i] = ((bins[i] <= uncert) & (uncert < bins[i + 1])).sum()
        avg_uncer[i] = uncert[(bins[i] <= uncert) & (uncert < bins[i + 1])].mean()

    # acc = [(labels[(b[0] <= uncert) & (uncert < b[1])] == arg_preds[(b[0] <= uncert) & (uncert < b[1])]).sum()
    #       for b in bins]
    # num = [((b[0] < uncert) & (uncert <= b[1])).sum() for b in bins]
    # avg_uncer = [uncert[((b[0] < uncert) & (uncert <= b[1]))].mean() for b in bins]
    # accs = [accs[i] + acc[i] for i in range(len(bins))]
    # nums = [nums[i] + num[i] for i in range(len(bins))]

    a = []
    for i in range(num_bins):

        if torch.isnan(avg_uncer[i]):
            avg_uncer[i] = torch.tensor([0.0])

        if nums[i] != 0:
            a.append(accs[i].float() / nums[i].float())
        else:
            a.append(torch.tensor([0.0]))

    a = torch.Tensor(a).to(nums.device)

    return a, nums, accs, avg_uncer


def computeECE(preds, label, num_bins, num_classes):
    preds = preds.detach()

    test_bin_val, test_nums, _, test_avg_uncer = calibrationBins(
        preds, compute_confidence(preds), label, num_bins, [1 / num_classes, 1]
    )

    test_bin_val = np.array(test_bin_val)
    test_nums = np.array(test_nums)
    # test_accs = np.array(test_accs)
    test_avg_uncer = np.array(test_avg_uncer)

    ece = np.sum(test_nums / np.sum(test_nums) * np.abs(test_bin_val - test_avg_uncer))

    return ece


def computeACE(preds, label, num_bins, num_classes):
    num_data_points = preds.size(0)

    tace = 0.0

    argm = torch.argmax(preds, 1)

    for c in range(num_classes):

        c_tace = 0.0

        c_label = label == c
        c_argm = argm == c

        c_acc = c_argm == c_label
        c_preds = preds[:, c]

        sort_ind = torch.argsort(c_preds, 0)

        c_preds = c_preds[sort_ind]
        c_acc = c_acc[sort_ind]

        chunk_c_preds = torch.chunk(c_preds, num_bins, 0)
        chunk_c_acc = torch.chunk(c_acc, num_bins, 0)

        for ch_preds, ch_acc in zip(chunk_c_preds, chunk_c_acc):
            size_chunk = ch_preds.size(0)

            c_tace += torch.abs(torch.mean(ch_preds) - torch.sum(ch_acc) / size_chunk) * (size_chunk / num_data_points)

        tace += c_tace * (1 / num_classes)

    return tace


# DISTANCE FUNCTIONS
def distFunc_predsDiff(preds1, preds2, labels):
    return torch.sum(torch.argmax(preds1, 1) != torch.argmax(preds2, 1))


def distFunc_kl(preds1, preds2, labels):
    return torch.sum(computeKL(preds1, preds2) + computeKL(preds2, preds1))


def distFunc_chi2(preds1, preds2, labels):
    return torch.sum(computeChiSquare(preds1, preds2))


def distFunc_rightwrong(preds1, preds2, labels):
    # Find out on how many predictions there will be an improvement
    arg_preds1 = torch.argmax(preds1)
    arg_preds2 = torch.argmax(preds2)

    correct_1 = arg_preds1 == labels
    correct_2 = arg_preds2 == labels

    sum_1 = torch.sum(torch.logical_and(torch.logical_not(correct_1), correct_2))
    sum_2 = torch.sum(torch.logical_and(correct_1, torch.logical_not(correct_2)))
    return sum_1 - sum_2


def diffMatrix(preds, ensembleIdList, compareLabels, metric="argmax_diff"):
    def argmax_diff(p1, p2, compareLabels):
        return torch.sum(torch.argmax(p1, 1) != torch.argmax(p2, 1)) / len(compareLabels)

    def doubleFault_diff(p1, p2, compareLabels):
        ind = torch.argmax(p1, 1) == torch.argmax(p2, 1)
        return torch.sum(torch.argmax(p1, 1)[ind] != compareLabels[ind])

    def Q_diff(p1, p2, compareLabels):

        p1_a = torch.argmax(p1, 1)
        p2_a = torch.argmax(p2, 1)

        same_ind = p1_a == p2_a
        diff_ind = p1_a != p2_a

        N00 = torch.sum(p1_a[same_ind] != compareLabels[same_ind])
        N11 = torch.sum(p1_a[same_ind] == compareLabels[same_ind])

        N10 = torch.sum(
            torch.logical_and(
                p1_a[diff_ind] == compareLabels[diff_ind],
                p2_a[diff_ind] != compareLabels[diff_ind],
            )
        )
        N01 = torch.sum(
            torch.logical_and(
                p1_a[diff_ind] != compareLabels[diff_ind],
                p2_a[diff_ind] == compareLabels[diff_ind],
            )
        )

        return (N00 * N11 - N10 * N01).float() / (N00 * N11 + N10 * N01).float()

    def kl_diff(p1, p2, compareLabels):
        return distFunc_kl(F.softmax(p1, 1), F.softmax(p2, 1), compareLabels)

    def chi2_diff(p1, p2, compareLabels):
        return distFunc_chi2(F.softmax(p1, 1), F.softmax(p2, 1), compareLabels)

    def weight_cos_diff(p1, p2, compareLabels):
        return torch.abs(torch.nn.functional.cosine_similarity(p1, p2, dim=0, eps=1e-8))

    implemented_metrics = {
        "argmax_diff": argmax_diff,
        "kl": kl_diff,
        "chi2": chi2_diff,
        "dF": doubleFault_diff,
        "Q": Q_diff,
        "weight_cos": weight_cos_diff,
    }

    metric = implemented_metrics[metric]

    if ensembleIdList is None:
        ensembleIdList = range(len(preds))
    else:
        preds = preds[ensembleIdList]

    matrix = np.zeros([len(ensembleIdList), len(ensembleIdList)])

    for id1 in ensembleIdList:
        for id2 in ensembleIdList:
            if id2 <= id1:
                continue

            val = metric(preds[id1], preds[id2], compareLabels)

            matrix[id1, id2] = val
            matrix[id2, id1] = val

    return matrix
