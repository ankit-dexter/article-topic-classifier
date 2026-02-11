import numpy as np

def compute_ece(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def coverage_accuracy_curve(probs, labels, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.99, 20)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)

    results = []

    for t in thresholds:
        mask = confidences >= t

        if np.sum(mask) == 0:
            coverage = 0.0
            acc = 0.0
        else:
            coverage = np.mean(mask)
            acc = np.mean(correct[mask])

        results.append((t, coverage, acc))

    return results