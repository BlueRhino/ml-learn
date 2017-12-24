import numpy as np


def train(train_data, learn_rate):
    # TODO:没有校验输入数据
    train_array = np.array(train_data)
    data_len = len(train_data)
    dimension = data_len - 1
    has_error = True
    w0 = 0.0
    b0 = 0.0
    while has_error:
        for i in range(data_len):
            x = train_array[i, 0:-1]
            yi = train_array[i, -1]
