import numpy as np
import idx2numpy
import scipy.io
from src.utils import unpickle


def load_mnist(dataset, dataset_path):

        
    if dataset == 'mnist':
        # Load files
        train_img = idx2numpy.convert_from_file(f'{dataset_path}/train-images.idx3-ubyte')
        train_labels = idx2numpy.convert_from_file(f'{dataset_path}/train-labels.idx1-ubyte')
        test_img = idx2numpy.convert_from_file(f'{dataset_path}/t10k-images.idx3-ubyte')
        test_labels = idx2numpy.convert_from_file(f'{dataset_path}/t10k-labels.idx1-ubyte')
        
        # Normalize the images
        train_img = ((np.expand_dims(train_img, axis=1) / 255) - 0.1307) / 0.3081
        test_img = ((np.expand_dims(test_img, axis=1) / 255) - 0.1307) / 0.3081
    elif dataset == 'fmnist':
        train_img = idx2numpy.convert_from_file(f'{dataset_path}/train-images-idx3-ubyte')
        train_labels = idx2numpy.convert_from_file(f'{dataset_path}/train-labels-idx1-ubyte')
        test_img = idx2numpy.convert_from_file(f'{dataset_path}/t10k-images-idx3-ubyte')
        test_labels = idx2numpy.convert_from_file(f'{dataset_path}/t10k-labels-idx1-ubyte')
        
        train_img = ((np.expand_dims(train_img, axis=1) / 255) - 0.2860) / 0.3530
        test_img = ((np.expand_dims(test_img, axis=1) / 255) - 0.2860) / 0.3530

    # Sample a validation set
    val_indices = np.load(f'val IDs/10000_val_indices_{dataset}.npy')
    # val_indices = np.random.choice(train_labels.shape[0], 10000, replace=False)
    val_img = train_img[val_indices]
    val_labels = train_labels[val_indices]
    train_img = np.delete(train_img, val_indices, axis=0)
    train_labels = np.delete(train_labels, val_indices, axis=0)

    return train_img, train_labels, val_img, val_labels, test_img, test_labels


def load_cifar(dataset, dataset_path):

    if dataset == 'cifar10':

        # Reference class values for min-max scaling
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]

        # Load data and apply global contrast optimization using the L1-norm
        train_img = np.empty((0, 3072))
        train_labels = []
        for i in range(1, 6):
            datadict = unpickle(f'{dataset_path}/data_batch_{str(i)}')
            train_img = np.concatenate((train_img, datadict[b'data']), axis=0)
            train_labels.extend(datadict[b'labels'])
        train_img -= train_img.mean(axis=1, keepdims=True)
        train_img /= np.abs(train_img).mean(axis=1, keepdims=True)
        train_img = train_img.reshape(train_img.shape[0], 3, 32, 32)
        train_labels = np.array(train_labels)

        datadict = unpickle(f'{dataset_path}/test_batch')
        test_img = datadict[b'data'].astype(float)
        test_img -= test_img.mean(axis=1, keepdims=True)
        test_img /= np.abs(test_img).mean(axis=1, keepdims=True)
        test_img = test_img.reshape(test_img.shape[0], 3, 32, 32)
        test_labels = np.array(datadict[b'labels'])

        del datadict

    elif dataset == 'cifar100':

        min_max = [(-14.475381165846034, 12.727414786250481), (-10.135715713633676, 12.160161817681036),
                   (-10.161284974396713, 14.850489500748308), (-19.3408532196652, 14.995088210139775),
                   (-9.367549800181296, 12.205577403284343), (-13.702793265179801, 14.258679008335188),
                   (-14.170852844849986, 10.851977970193277), (-15.71542712113897, 9.566342204557307),
                   (-7.860766655986478, 8.843353910552612), (-5.972078245203766, 10.153849282936006),
                   (-9.781021309203902, 8.556866284005276), (-7.734030335714939, 10.450245213389953),
                   (-10.214244398024642, 9.35366007307547), (-18.856031884412687, 21.160188606242144),
                   (-8.212590935820346, 10.639574374656794), (-12.311221417322837, 10.493695809835845),
                   (-7.833277239891915, 10.35800168761534), (-6.648282524262063, 11.645584049487276),
                   (-7.180139184279749, 11.97570811709053), (-9.808531530992028, 10.469580417662428)]

        # Load data and apply global contrast optimization using the L1-norm
        trainData = unpickle(f'{dataset_path}/train')
        testData = unpickle(f'{dataset_path}/test')

        train_img = np.array(trainData[b'data']).astype(float)
        train_img -= train_img.mean(axis=1, keepdims=True)
        train_img /= np.abs(train_img).mean(axis=1, keepdims=True)
        train_img = train_img.reshape(train_img.shape[0], 3, 32, 32)
        train_labels = np.array(trainData[b'coarse_labels'])

        test_img = np.array(testData[b'data']).astype(float)
        test_img -= test_img.mean(axis=1, keepdims=True)
        test_img /= np.abs(test_img).mean(axis=1, keepdims=True)
        test_img = test_img.reshape(test_img.shape[0], 3, 32, 32)
        test_labels = np.array(testData[b'coarse_labels'])

        del trainData, testData

    # Sample a validation set
    val_indices = np.load(f'val IDs/10000_val_indices_{dataset}.npy')
    # val_indices = np.random.choice(train_labels.shape[0], 10000, replace=False)
    val_img = train_img[val_indices]
    val_labels = train_labels[val_indices]
    train_img = np.delete(train_img, val_indices, axis=0)
    train_labels = np.delete(train_labels, val_indices, axis=0)

    return train_img, train_labels, val_img, val_labels, test_img, test_labels, min_max


def load_arrhythmia(dataset_path):

    # Load files
    mat = scipy.io.loadmat(f'{dataset_path}/arrhythmia.mat')
    X = mat['X']
    y = mat['y']

    # Separate normal and anomalous samples
    Xa = X[y.reshape(y.shape[0]).astype(bool), :]
    Xn = X[np.logical_not(y.reshape(y.shape[0]).astype(bool)), :]

    # Sample from normal class for val and test
    indices = np.load('val IDs/randomized_normal_indices_arrhythmia.npy')
    # indices = np.random.choice(Xn.shape[0], Xn.shape[0], replace=False)
    num_samples = indices.shape[0]
    Xn_test = Xn[indices[:np.round(0.2 * num_samples).astype(int)]]
    Xn_val = Xn[indices[np.round(0.2 * num_samples).astype(int):np.round((0.4) * num_samples).astype(int)]]

    X_train = Xn[indices[np.round((0.4) * num_samples).astype(int):]]

    # Sample from anomalous class for val and test
    indices = np.load('val IDs/randomized_anomalous_indices_arrhythmia.npy')
    # indices = np.random.choice(Xa.shape[0], Xa.shape[0], replace=False)
    num_samples = indices.shape[0]
    Xa_test = Xa[indices[:np.round(0.2 * indices.shape[0]).astype(int)]]
    Xa_val = Xa[indices[np.round(0.2 * num_samples).astype(int):np.round((0.4) * num_samples).astype(int)]]
    X_test = np.concatenate((Xn_test, Xa_test), axis=0)
    X_val = np.concatenate((Xn_val, Xa_val), axis=0)

    y_test = np.concatenate((np.zeros(Xn_test.shape[0]), np.ones(Xa_test.shape[0])), axis=0).astype(int)
    y_val = np.concatenate((np.zeros(Xn_val.shape[0]), np.ones(Xa_val.shape[0])), axis=0).astype(int)

    return X_train, X_val, X_test, y_val, y_test
