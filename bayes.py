import numpy as np
from timeit import default_timer
import random

feature_size = 4
parts = 28 // feature_size
num_trained_image = 500

def count_pixel(matrix):
    black = 0
    white = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                white += 1
            else:
                black += 1
    return black, white

def predict(data_test, L, appearence_record):
    data_test = np.array(data_test)
    prob = {}
    for i in range(10):
        prob[i] = 1
    for i in range(len(L)):
        for j in range(len(L[i])):
            row_num = int(j / parts)
            column_num = j % parts
            matrix = data_test[row_num * feature_size:(row_num * feature_size + feature_size),
                     column_num * feature_size:(column_num * feature_size + feature_size)]
            black_num,white_num = count_pixel(matrix)
            prob[i] *= (0.001 if L[i][j][0][white_num] == 0 else L[i][j][0][white_num])
            prob[i] *= (0.001 if L[i][j][1][black_num] == 0 else L[i][j][1][black_num])
    for i in range(10):
        prob[i] *= appearence_record[i]/num_trained_image
    max_key = max(prob, key=prob.get)
    return max_key


if __name__ == "__main__":
    directory = './extracted_digit_training_data'

    # L = [[[[0 for z in range(feature_size ** 2+1)] for h in range(2)] for j in range(parts ** 2)] for i in range(10)]

    labels = []
    with open('./digitdata/traininglabels') as f:
        for line in f:
            labels.append(int(line))
    # print(labels[100])

    labels_test = []
    with open('./digitdata/testlabels') as f:
        for line in f:
            labels_test.append(int(line))

    datas = []
    for f in range(5000):
        data = np.genfromtxt(directory + '/digit_%d' % f, dtype=float)
        datas.append(data)
    datas = np.array(datas)

    # training
    for epoch in range(5):
        L = [[[[0 for z in range(feature_size ** 2+1)] for h in range(2)] for j in range(parts ** 2)] for i in range(10)]

        start_time = default_timer()
        random_indexes = random.sample([i for i in range(5000)],num_trained_image)
        appearence_record = [0 for i in range(10)]
        for f in range(len(random_indexes)):
            # divide into parts
            appearence_record[labels[random_indexes[f]]] += 1
            for i in range(parts ** 2):
                row_num = int(i / parts)
                column_num = i % parts
                # first is label, second is part of matrix
                matrix = datas[random_indexes[f]][row_num * feature_size:(row_num * feature_size + feature_size),
                         column_num * feature_size:(column_num * feature_size + feature_size)]
                black_num,white_num = count_pixel(matrix)

                #       black = 1, white = 0
                L[labels[random_indexes[f]]][i][1][black_num] += 1/num_trained_image
                L[labels[random_indexes[f]]][i][0][white_num] += 1/num_trained_image

        end_time =default_timer()
        print(round(end_time-start_time,3),"s")

        # predict
        true_count = 0
        for f in range(1000):
            data_test = np.genfromtxt("./extracted_digit_test_data/digit_%d" % f, dtype=float)
            predict_num = predict(data_test, L, appearence_record)
            # print(predict_num,labels_test[f])
            if predict_num == labels_test[f]:
                true_count += 1
        accuracy = true_count / len(labels_test)
        print(accuracy)
