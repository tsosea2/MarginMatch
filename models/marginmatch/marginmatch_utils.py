import torch
import math
import torch.nn.functional as F
import numpy as np

from train_utils import ce_loss

class AUMCalculator:

    def __init__(self, margin_smoothing, num_labels, num_examples, percentile) -> None:
        self.delta = margin_smoothing

        self.num_labels = num_labels
        self.num_examples = num_examples

        self.AUMMatrix = {}
        self.t = {}

        self.threshold_aum_examples = {}
        self.threshold_t = {}
        
        self.percentile = percentile

        for i in range(num_examples):
            self.AUMMatrix[i] = np.zeros(num_labels)
            self.t[i] = 0

    def get_aums(self, ids):
        x = []
        for id in ids:
            if id not in self.threshold_aum_examples:
                x.append(self.AUMMatrix[id])
            else:
                x.append(self.threshold_aum_examples[id])

        return np.array(x)

    def switch_threshold_examples(self, ids):

        self.threshold_aum_examples = {}
        self.threshold_t = {}
        for id in ids:
            self.threshold_aum_examples[id] = np.ones(self.num_labels) * 0
            self.threshold_t[id] = 0
        self.num_threshold_examples = len(ids)

    def retrieve_threshold(self):
        if self.num_threshold_examples == 0:
            return 0
        
        threshold_pool = []
        for threshold_example in self.threshold_aum_examples:
            threshold_pool.append(self.threshold_aum_examples[threshold_example][-1])
        threshold_pool.sort(reverse=True)
        print(threshold_pool)
        return threshold_pool[int((self.num_threshold_examples * self.percentile) // 100)]

    def update_aums(self, ids, margins):
        for i in range(len(ids)):
            if ids[i] not in self.threshold_aum_examples:
                self.AUMMatrix[ids[i]] = margins[i] * self.delta / \
                    (1 + self.t[ids[i]]) + self.AUMMatrix[ids[i]] * \
                    (1 - self.delta / (1 + self.t[ids[i]]))
                self.t[ids[i]] += 1
            else:
                self.threshold_aum_examples[ids[i]] = margins[i] * self.delta / \
                    (1 + self.threshold_t[ids[i]]) + self.threshold_aum_examples[ids[i]] * \
                    (1 - self.delta / (1 + self.threshold_t[ids[i]]))
                self.threshold_t[ids[i]] += 1

class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value


def consistency_loss(logits_s, logits_w, conf_acc, p_target, p_model, x_ulb_idx, aum_calculator, y_ulb, crt_threshold, num_classes, threshold_mask, name='ce',
                     T=1.0, p_cutoff=0.0, use_hard_labels=True, use_DA=False, labels=True):

    assert name in ['ce', 'L2']

    logits_w = logits_w.detach()

    max_logits = torch.max(logits_w, dim=-1)
    mask = logits_w != max_logits.values[:, None]
    partial = logits_w - mask * max_logits.values[:, None]
    second_largest = torch.max(mask * logits_w, dim=-1)
    second_largest = ~mask * second_largest.values[:, None]
    margins = partial - second_largest

    ids = x_ulb_idx.detach().cpu().numpy()
    aum_calculator.update_aums(ids, margins.cpu().numpy())

    crt_aums = aum_calculator.get_aums(ids)
    crt_aums = crt_aums[np.arange(logits_w.shape[0]), max_logits.indices.cpu()]
    crt_aums = torch.tensor(crt_aums).cuda()


    pseudo_label = torch.softmax(logits_w[:, :-1], dim=-1)
    if use_DA:
        if p_model == None:
            p_model = torch.mean(pseudo_label.detach(), dim=0)
        else:
            p_model = p_model * 0.999 + \
                torch.mean(pseudo_label.detach(), dim=0) * 0.001
        pseudo_label = pseudo_label * p_target / p_model
        pseudo_label = (pseudo_label / pseudo_label.sum(dim=-1, keepdim=True))

    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(
        p_cutoff * (conf_acc[max_idx] / (2. - conf_acc[max_idx]))).float()

    if labels is True:
        confidence_acc = sum(mask * (max_idx == y_ulb)) / sum(mask)
    else:
        confidence_acc = None

    if crt_threshold is None:
        both_mask = mask
        confidence_aum = confidence_acc
    else:
        aum_mask = crt_aums.ge(crt_threshold).float()
        confidence_aum = sum(aum_mask * (max_idx == y_ulb)) / sum(aum_mask)
        both_mask = mask * aum_mask

    if labels is True:
        confidence_aum_acc = sum(
            both_mask * (max_idx == y_ulb)) / sum(both_mask)
    else:
        confidence_aum_acc = None

    inverse_mask = (~threshold_mask).int()
    max_idx = max_idx * inverse_mask + threshold_mask * (num_classes - 1)

    saved_both_mask = both_mask
    both_mask = both_mask * inverse_mask + threshold_mask.float() * (sum(saved_both_mask) / logits_w.shape[0])


    select_confidence = max_probs.ge(p_cutoff).long()

    select_confidence = select_confidence * inverse_mask

    if crt_threshold is not None:
        select_aum = crt_aums.ge(crt_threshold).long()
    else:
        select_aum = select_confidence

    masked_loss = ce_loss(logits_s, max_idx, use_hard_labels,
                          reduction='none') * both_mask

    mask = torch.tensor(mask, dtype=torch.float)

    if crt_threshold is not None:
        aum_mask = torch.tensor(aum_mask, dtype=torch.float)
    else:
        aum_mask = mask

    return masked_loss.mean(), mask.mean(), aum_mask.mean(), select_confidence, select_aum, max_idx.long(), p_model, confidence_acc, confidence_aum_acc, confidence_aum, sum(saved_both_mask)
