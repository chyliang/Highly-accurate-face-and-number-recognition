import numpy as np
from timeit import default_timer
import random

white_pixel_point = 0
black_pixel_point = 1
num_of_samples = 5000
looping_time = 0

if __name__ == "__main__":

    # y
    labels = []
    with open('./digitdata/traininglabels') as f:
        for line in f:
            labels.append(int(line))

    directory = './extracted_digit_training_data'
    training_set = []
    # get x, and construct training set
    for f in range(5000):
        data = np.genfromtxt(directory + '/digit_%d' % f, dtype=float)
        training_set.append((data, labels[f]))

    labels_test = []
    with open('./digitdata/testlabels') as f:
        for line in f:
            labels_test.append(int(line))

    for epoch in range(5):
        w = [[0 for j in range(28 * 28 + 1)] for i in range(10)]
        w = np.array(w)
        random_indexes = random.sample([i for i in range(5000)], num_of_samples)
        start_time = default_timer()
        while (True):
            for i in range(len(random_indexes)):
                score = [0 for h in range(10)]
                for j in range(10):
                    for z in range(len(w[0]) - 1):
                        row_num = int(z / 28)
                        column_num = z % 28
                        score[j] += w[j][z + 1] * (
                            black_pixel_point if training_set[random_indexes[i]][0][row_num][
                                                     column_num] == 1.0 else white_pixel_point)
                predict_y = np.argmax(np.array(score))
                if predict_y != training_set[random_indexes[i]][1]:
                    for z in range(len(w[predict_y]) - 1):
                            row_num = int(z / 28)
                            column_num = z % 28
                            w[predict_y][z + 1] = w[predict_y][z + 1] - (
                                white_pixel_point if training_set[random_indexes[i]][0][row_num][
                                                         column_num] == 0 else black_pixel_point)
                            w[training_set[random_indexes[i]][1]][z + 1] = w[training_set[random_indexes[i]][1]][z + 1] + (
                                            white_pixel_point if training_set[random_indexes[i]][0][row_num][
                                                                     column_num] == 0 else black_pixel_point)
            end_time = default_timer()
            if end_time - start_time > looping_time:
                print(round(end_time - start_time, 3), "s")
                break
        # with open('weight_%s.txt' % str(num_of_samples), 'w') as out_file:
        #     json.dump(w.tolist(), out_file)

        count = 0
        for f in range(1000):
            data_test = np.genfromtxt("./extracted_digit_test_data/digit_%d" % f, dtype=float)
            score = [0 for h in range(10)]
            for j in range(len(w)):
                for z in range(len(w[0]) - 1):
                    row_num = int(z / 28)
                    column_num = z % 28
                    score[j] += (white_pixel_point if data_test[row_num][column_num] == 0 else black_pixel_point) * \
                                w[j][z + 1]
            predict_y = np.argmax(np.array(score))
            if predict_y == labels_test[f]:
                count += 1
        print(count / 1000)
