import os
import json
import math
import torch
import numpy as np
import pandas as pd
import argparse
from TimeSeries import TimeSeries
import timeit
import wrappers_original
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from pyts.datasets import fetch_uea_dataset
from pathlib import Path

def load_UEA_dataset(path, dataset, train_ratio=0.9,random_state=0):
    """
    Loads the UEA dataset given in input in np arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    # Initialization needed to load a file with Weka wrappers_test
    try:
        train, test, train_labels, test_labels = fetch_uea_dataset(dataset, return_X_y=True)
        le = LabelEncoder()
        train_labels = le.fit_transform(train_labels)
        test_labels = le.transform(test_labels)
    except:
        trainTS = {}
        testTS = {}
        
        train_raw = pd.read_csv((path + "/" + dataset + "/" + dataset + "_TRAIN.txt"), delim_whitespace=True, header=None)
        test_raw = pd.read_csv((path + "/" + dataset + "/" + dataset + "_TEST.txt"), delim_whitespace=True, header=None)

        for i in range(train_raw.shape[0]):
            label = int(train_raw.iloc[i, 0])
            series = train_raw.iloc[i,1:].tolist()
            trainTS[i] = TimeSeries(series, label)
            trainTS[i].NORM(True)

        for i in range(test_raw.shape[0]):
            label = int(test_raw.iloc[i, 0])
            series = test_raw.iloc[i, 1:].tolist()
            testTS[i] = TimeSeries(series, label)
            testTS[i].NORM(True)

        train = np.array([np.array(trainTS[i].data) for i in range(len(trainTS))])
        train = train.reshape(train.shape[0], 1, train.shape[1])
        train_labels = np.array([np.array(trainTS[i].label) for i in range(len(trainTS))])
        
        test = np.array([np.array(testTS[i].data) for i in range(len(testTS))])
        test = test.reshape(test.shape[0], 1, test.shape[1])
        test_labels = np.array([np.array(testTS[i].label) for i in range(len(testTS))])

    if train_ratio < 1:
        X_train_ori, y_train_ori = train, train_labels
        sss = StratifiedShuffleSplit(n_splits=10, test_size=1 - train_ratio, random_state=random_state)
        sss.get_n_splits(X_train_ori, y_train_ori)

        for train_index, test_index in sss.split(X_train_ori, y_train_ori):
            train = X_train_ori[train_index,:]
            train_labels = y_train_ori[train_index]
    
        print(f'Ratio {train_ratio} - train shape: {np.shape(train)}')
    print(f'dataset load succeed for random state {random_state}!!!')
    return train, train_labels, test, test_labels


def fit_parameters(file, ratio, train, train_labels, test, test_labels, dataset, cuda, gpu, save_path, cluster_num, random_state,
                        save_memory=False):
    """
    Creates a classifier from the given set of parameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = wrappers_original.CausalCNNEncoderClassifier()

    # Loads a given set of parameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    params['in_channels'] = 1
    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)
    return classifier.fit(
        ratio, train, train_labels, test, test_labels, dataset, save_path, cluster_num, random_state, save_memory=save_memory, verbose=True
    )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of parameters to use ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the classifier')
    parser.add_argument('--ratio', type=float, default=1,
                        help='percent of training samples used for few-shot learning')

    print('parse arguments succeed !!!')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if not Path(args.save_path).exists():
        Path(args.save_path).mkdir(parents=True)
    csv_file = Path(str(args.save_path), str(args.dataset) + '.csv')
    if csv_file.exists():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=['ratio', 'random_state', 'accuracy'])
        df.to_csv(csv_file, index=False)
    
    for random_state in range(3):
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]:
            if ratio == 1 & random_state != 0:
                continue
            try:
                if not df[(df['ratio'] == ratio) & (df['random_state'] == random_state)].empty:
                    print('Already done')
                    continue
                start = timeit.default_timer()
                if args.cuda and not torch.cuda.is_available():
                    print("CUDA is not available, proceeding without it...")
                    args.cuda = False

                train, train_labels, test, test_labels = load_UEA_dataset(
                    args.path, args.dataset, ratio, random_state
                )
                cluster_num = 100
                if not args.load and not args.fit_classifier:
                    print('start new network training')
                    classifier = fit_parameters(
                    args.hyper, ratio, train, train_labels, test, test_labels, args.dataset, args.cuda, args.gpu, args.save_path, cluster_num, random_state
                    )

                if not args.load:
                    if args.fit_classifier:
                        classifier.fit_classifier(classifier.encode(train), train_labels)
                    classifier.save(
                        os.path.join(args.save_path, args.dataset), ratio, random_state
                    )
                    with open(
                        os.path.join(
                            args.save_path, args.dataset + '_parameters.json'
                        ), 'w'
                    ) as fp:
                        json.dump(classifier.get_params(), fp)

                end = timeit.default_timer()
                print(f"All time for ratio {ratio} random_state {random_state}: ", (end- start)/60)
            except Exception as e:
                print('ratio {} random_state {} failed'.format(ratio, random_state))
                print(e)
                continue

