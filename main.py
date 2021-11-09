from driftdetector.DriftDetector import DriftDetector


def main():
    import csv
    import numpy as np

    device = 'cpu'

    training_window_size = 100
    test_batch_size = 4
    steps_generator = 50
    epochs = 50
    equalize = True
    sequence_length = 1
    batch_size = 8
    generator_batch_size = 2
    lr = 1.0

    dd = DriftDetector(training_window_size=training_window_size, device=device, epochs=epochs, batch_size=batch_size,
                       equalize=equalize, sequence_length=sequence_length, lr=lr,
                       steps_generator=steps_generator, generator_batch_size=generator_batch_size)

    path = '../driftdetection/datasets/real-world/outdoorStream.csv'
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
    mean = np.mean(features, axis=1).reshape(features.shape[0], 1)
    std = np.std(features, axis=1).reshape(features.shape[0], 1)
    features = (features - mean) / (std + 0.000001)
    dd.initialize_detection_model(features=features[:training_window_size])
    idx = training_window_size
    while idx + training_window_size < len(features):
        detected = dd.detect_drifts(features[idx:idx+test_batch_size])
        if not detected:
            idx += test_batch_size
            continue

        print('Drift detected at index %d' % idx)
        # Collect data until window is full and then retrain the model
        data = features[idx:idx+training_window_size]
        dd.retrain_model(old_features=features[idx-training_window_size, :],
                         new_features=features[idx+training_window_size, :])
        idx += training_window_size

    print(idx)
    print('Done')


if __name__ == '__main__':
    main()
