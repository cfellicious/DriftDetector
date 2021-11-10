from driftdetector.DriftDetector import DriftDetector
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def fit_and_predict(clf, features, labels, classes):
    predicted = np.empty(shape=len(labels))
    predicted[0] = clf.predict([features[0]])
    clf.reset()
    clf.partial_fit([features[0]], [labels[0]], classes=classes)
    for idx in range(1, len(labels)):
        predicted[idx] = clf.predict([features[idx]])
        clf.partial_fit([features[idx]], [labels[idx]], classes=classes)

    return predicted, clf


def predict_and_partial_fit(clf, features, labels, classes):
    predicted = np.empty(shape=len(labels))
    for idx in range(0, len(labels)):
        predicted[idx] = clf.predict([features[idx]])
        clf.partial_fit([features[idx]], [labels[idx]], classes=classes)

    return predicted, clf


def main():
    import csv
    import numpy as np

    device = 'cuda'

    training_window_size = 100
    test_batch_size = 8
    steps_generator = 50
    epochs = 50
    equalize = True
    sequence_length = 1
    batch_size = 8
    generator_batch_size = 2
    lr = 1.0

    y_pred = []
    y_true = []
    clf = HoeffdingTreeClassifier()

    dd = DriftDetector(training_window_size=training_window_size, device=device, epochs=epochs, batch_size=batch_size,
                       equalize=equalize, sequence_length=sequence_length, lr=lr,
                       steps_generator=steps_generator, generator_batch_size=generator_batch_size)

    path = '../driftdetection/datasets/real-world/airlines2.csv'
    array = []
    labels = []

    with open(path, 'r') as f:
        lines = csv.reader(f)
        # Skip headers
        next(lines)
        for line in lines:
            temp = [float(x) for x in line]
            labels.append(int(temp[-1]))
            temp = temp[:-1]
            array.append(temp)

    features = np.array(array)
    backup_features = np.copy(features, copy=True)
    mean = np.mean(features, axis=1).reshape(features.shape[0], 1)
    std = np.std(features, axis=1).reshape(features.shape[0], 1)
    features = (features - mean) / (std + 0.000001)
    dd.initialize_detection_model(features=features[:training_window_size])
    idx = training_window_size

    classes = np.unique(labels)
    x = features[:training_window_size, :]
    y = labels[:training_window_size]

    predicted, clf = fit_and_predict(clf=clf, features=x, labels=y, classes=classes)
    y_pred = y_pred + predicted.tolist()
    y_true = y_true + y

    while idx + training_window_size < len(features):
        detected = dd.detect_drifts(features[idx:idx+test_batch_size])
        if not detected:
            idx += test_batch_size
            predicted, clf = predict_and_partial_fit(clf=clf, features=features[idx:idx+test_batch_size],
                                                     labels=labels[idx:idx+test_batch_size],
                                                     classes=classes)
            y_pred = y_pred + predicted.tolist()
            y_true = y_true + labels[idx:idx+test_batch_size]
            continue

        print('Drift detected at index %d' % idx)
        # Collect data until window is full and then retrain the model
        data = features[idx:idx+training_window_size]
        dd.retrain_model(old_features=features[idx-training_window_size:idx, :],
                         new_features=features[idx:idx+training_window_size, :])

        # Drift detected, so retrain classifier
        training_idx_start = idx
        training_idx_end = idx + training_window_size
        # retrain the classifier because of partial drift.

        predicted, clf = fit_and_predict(clf=clf, features=features[training_idx_start:training_idx_end, :],
                                         labels=labels[training_idx_start:training_idx_end],
                                         classes=classes)
        predicted = predicted.tolist()
        y_pred = y_pred + predicted
        y_true = y_true + labels[training_idx_start:training_idx_end]

        idx += training_window_size

    auc_value = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(auc_value)
    print(idx)
    print('Done')


if __name__ == '__main__':
    main()
