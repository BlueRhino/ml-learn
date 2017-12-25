import numpy as np
from numpy import where
from sklearn.datasets import make_classification


def train(train_data, learn_rate):
    # TODO:没有校验输入数据，也没有确定训练集是线程可分的
    train_array = np.array(train_data)
    data_len = len(train_data)
    dimension = len(train_data[0]) - 1
    has_error = True
    w = np.zeros(dimension)
    b = 0.0
    learn_times = 0
    while has_error:
        learn_times += 1
        has_error = False
        for i in range(data_len):
            x = train_array[i, 0:-1]
            yi = train_array[i, -1]
            res = inner_function(w, x, b)
            inner_res = yi * res
            if inner_res <= 0:
                w = w + learn_rate * yi * x
                b = b + learn_rate * yi
                has_error = True
        plot(train_data, w, b, learn_times)
    return w, b


def inner_function(w, x, b):
    return np.inner(w, x) + b


def plot(train_data, w, b, label):
    # TODO:只能画二维图，并且没有判断输入数据的正确性
    import matplotlib.pyplot as plt
    plt.clf()
    train_data = np.array(train_data)
    positive_point_index_arr = where(train_data[:, -1] == 1)
    opposite_point_index_arr = where(train_data[:, -1] == -1)
    for tmp in positive_point_index_arr[0]:
        plt.scatter(train_data[tmp][0], train_data[tmp][1], c='r', marker='.')
    for tmp in opposite_point_index_arr[0]:
        plt.scatter(train_data[tmp][0], train_data[tmp][1], c='b', marker='.')
    x1 = np.array([-5, 5])
    x2 = ((-b) - w[0] * x1) / w[1]
    # plt.plot(x1, x2, label=label)
    plt.plot(x1, x2)
    # plt.legend()
    plt.savefig('../img/pla/' + str(label) + '.png')
    plt.show()


def generate_test_dataset(must_linear_separable):
    samples = []
    if must_linear_separable:
        separable = False
        while not separable:
            samples = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                          n_informative=1,
                                          n_clusters_per_class=1, flip_y=-1)
            # samples = make_classification
            # (n_samples=50, n_features=2, n_redundant=0, n_informative=1,
            #                               n_clusters_per_class=1, flip_y=-1, class_sep=1.5)
            red = samples[0][samples[1] == 0]
            blue = samples[0][samples[1] == 1]
            separable = any(
                [red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in
                 range(2)])
    else:
        samples = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1,
                                      n_clusters_per_class=1, flip_y=-1, class_sep=1.5)
    return samples


def main():
    train_data = []
    samples = generate_test_dataset(False)
    for index, label in enumerate(samples[1]):
        if label > 0:
            train_data.append(np.insert(samples[0][index], len(samples[0][index]), values=1))
        else:
            train_data.append(np.insert(samples[0][index], len(samples[0][index]), values=-1))
    l_rate = 1
    w, b = train(train_data, l_rate)
    print(w)
    print(b)


if __name__ == '__main__':
    main()
