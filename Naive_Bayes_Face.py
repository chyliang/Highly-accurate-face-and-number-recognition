import numpy as np
import random
from timeit import default_timer

dividing_row_len = 4
divided_parts = 28 // dividing_row_len
num_of_samples = int(451*1)


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


def train(faces, labels, features, random_indexes):
    # num_faces_true = len([z for z in labels if z == 1])
    for j in range(len(random_indexes)):
        for i in range(int(divided_parts ** 2)):
            row_num = int(i / divided_parts)
            col_num = i % divided_parts
            black_num, white_num = count_pixel(
                faces[random_indexes[j]][row_num * dividing_row_len:(row_num * dividing_row_len + dividing_row_len),
                col_num * dividing_row_len:(col_num * dividing_row_len + dividing_row_len)])
            # feature[0] is black, feature[1] is white
            # features[0][i][black_num][1] is face, features[0][i][black_num][0] is not face
            if labels[random_indexes[j]] == 1:
                features[0][i][black_num][1] += (1 / len(faces))
                features[1][i][white_num][1] += (1 / len(faces))
            else:
                features[0][i][black_num][0] += (1 / len(faces))
                features[1][i][white_num][0] += (1 / len(faces))
    return features


def predict(face, features, appearence_record):
    black_list = []
    white_list = []
    for i in range(int(divided_parts ** 2)):
        row_num = int(i / divided_parts)
        col_num = i % divided_parts
        black_num, white_num = count_pixel(
            face[row_num * dividing_row_len:(row_num * dividing_row_len + dividing_row_len),
            col_num * dividing_row_len:(col_num * dividing_row_len + dividing_row_len)])
        black_list.append(black_num)
        white_list.append(white_num)
    prob_true = 1
    prob_false = 1
    for i in range(len(black_list)):
        prob_true *= (0.0001 if features[0][i][black_list[i]][1] == 0 else features[0][i][black_list[i]][1])
        prob_false *= (0.0001 if features[0][i][black_list[i]][0] == 0 else features[0][i][black_list[i]][0])
        prob_true *= (0.0001 if features[1][i][white_list[i]][1] == 0 else features[1][i][white_list[i]][1])
        prob_false *= (0.0001 if features[1][i][white_list[i]][0] == 0 else features[1][i][white_list[i]][0])

    prob_true *= appearence_record[1] / num_of_samples
    prob_false *= appearence_record[0] / num_of_samples

    if prob_true >= prob_false:
        return True
    else:
        return False


if __name__ == "__main__":

    faces = []
    for f in range(451):
        data = np.genfromtxt('./extracted_face_training_data_face/face_%d' % f, dtype=float)
        faces.append(data)
    faces = np.array(faces)

    labels = []
    # [0] is false, 1 is true
    appearence_record = [0 for i in range(2)]
    with open('./facedata/facedatatrainlabels') as f:
        for line in f:
            labels.append(int(line))
            if int(line) == 1:
                appearence_record[1] += 1
            else:
                appearence_record[0] += 1

    face_test = []
    for f in range(150):
        data = np.genfromtxt('./extracted_face_test_data/face_%d' % f, dtype=float)
        face_test.append(data)
    face_test = np.array(face_test)

    labels_test = []
    with open('./facedata/facedatatestlabels') as f:
        for line in f:
            labels_test.append(int(line))

    for i in range(5):
        # first is black or white, second is parts, third is how many white or black it has
        features = [
            [[[0 for h in range(2)] for z in range(int(dividing_row_len) ** 2 + 1)] for i in
             range(int(divided_parts ** 2))]
            for
            j in range(2)]
        random_indexes = random.sample([i for i in range(451)], num_of_samples)
        start_time = default_timer()
        features = train(faces, labels, features, random_indexes)

        end_time = default_timer()
        print(round(end_time - start_time, 3), "s")

        count = 0
        for f in range(len(labels_test)):
            result = predict(face_test[f], features, appearence_record)
            if ((result == True) and (labels_test[f] == 1)) or ((result == False) and (labels_test[f] == 0)):
                count += 1
        print(round(count / len(labels_test),3))
