import os

import numpy as np

import provider


def load_data(train_files,
              test_files,
              num_points=1024,
              shuffle=False,
              rotate=False,
              rotate_val=False):
    data = []
    label = []

    train_file_num = np.arange(len(train_files))

    for file_num in train_file_num:
        current_data, current_label = provider.loadDataFile(
            train_files[file_num])
        current_data = current_data[:, :num_points, :]

        if shuffle:
            current_data, current_label, _ = provider.shuffle_data(
                current_data, np.squeeze(current_label))
            current_label = np.expand_dims(current_label, axis=-1)

        data.append(current_data)
        label.append(current_label)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)

    test_data = []
    test_label = []

    test_file_num = np.arange(len(test_files))

    for file_num in test_file_num:
        current_data, current_label = provider.loadDataFile(
            train_files[file_num])
        current_data = current_data[:, :num_points, :]

        if shuffle:
            current_data, current_label, _ = provider.shuffle_data(
                current_data, np.squeeze(current_label))
            current_label = np.expand_dims(current_label, axis=-1)

        test_data.append(current_data)
        test_label.append(current_label)

    test_data = np.concatenate(test_data, axis=0)
    test_label = np.concatenate(test_label, axis=0)

    if rotate:
        data = rotate_point_cloud(data)
    if rotate_val:
        test_data = rotate_point_cloud(test_data)

    return (data, label), (test_data, test_label)


def rotate_data(files,
                num_points=1024,
                rotate=False):

    data = []
    label = []

    file_num = np.arange(len(files))

    for file_num in file_num:
        current_data, current_label = provider.loadDataFile(files[file_num])
        current_data = current_data[:, :num_points, :]

        data.append(current_data)
        label.append(current_label)

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)

    if rotate:
        data = rotate_point_cloud(data)

    return (data, label)


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based about z-axis
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0.],
                                    [-sinval, cosval, 0.],
                                    [0., 0., 1.]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(
            shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


if __name__ == "__main__":
    train_files = provider.getDataFiles(
        './data/modelnet40_ply_hdf5_2048/train_files.txt')
    test_files = provider.getDataFiles(
        './data/modelnet40_ply_hdf5_2048/test_files.txt')

    (x_train, y_train), (x_test, y_test) = load_data(train_files, test_files)
