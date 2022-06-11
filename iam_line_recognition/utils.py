import torch
import fastwer
import numpy as np
from scipy.special import logsumexp


"""
-------------
 CTC decoder
-------------
"""

NINF = -1 * float("inf")
DEFAULT_EMISSION_THRESHOLD = 0.01

def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # delete blank
    new_labels = [l for l in new_labels if l != blank]
    return new_labels

def beam_search_decode(emission_log_prob, blank=0, **kwargs):
    beam_size = kwargs["beam_size"]
    emission_threshold = kwargs.get("emission_threshold", np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                # log(p1 * p2) = log_p1 + log_p2
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        # sorted by accumulated_log_prob
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # sum up beams to produce labels
    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix, blank))
        # log(p1 + p2) = logsumexp([log_p1, log_p2])
        total_accu_log_prob[labels] = \
            logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    return labels

def greedy_decode(emission_log_prob, blank=0):
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)
    return labels

def ctc_decode(log_probs, which_ctc_decoder="beam_search", label_2_char=None, blank=0, beam_size=25):
    emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoded_list = []
    for emission_log_prob in emission_log_probs:
        if which_ctc_decoder == "beam_search":
            decoded = beam_search_decode(emission_log_prob, blank=blank, beam_size=beam_size)
        elif which_ctc_decoder == "greedy":
            decoded = greedy_decode(emission_log_prob, blank=blank)
        else:
            print(f"unidentified option for which_ctc_decoder : {which_ctc_decoder}")
            sys.exit(0)

        if label_2_char:
            decoded = [label_2_char[l] for l in decoded]
        decoded_list.append(decoded)
    return decoded_list

"""
--------------------
 Evaluation Metrics
--------------------
"""
def compute_wer_and_cer_for_batch(batch_preds, batch_gts):
    cer_batch = fastwer.score(batch_preds, batch_gts, char_level=True)
    wer_batch = fastwer.score(batch_preds, batch_gts)
    return cer_batch, wer_batch

def compute_wer_and_cer_for_sample(str_pred, str_gt):
    cer_sample = fastwer.score_sent(str_pred, str_gt, char_level=True)
    wer_sample = fastwer.score_sent(str_pred, str_gt)
    return cer_sample, wer_sample
