import numpy as np
from timeit import default_timer
import random

looping_time = 0
white_pixel_point = 0
black_pixel_point = 1
num_of_samples = int(451*1)

if __name__ == "__main__":

    faces = []
    for f in range(451):
        data = np.genfromtxt('./extracted_face_training_data_face/face_%d' % f, dtype=float)
        faces.append(data)
    faces = np.array(faces)

    labels = []
    with open('./facedata/facedatatrainlabels') as f:
        for line in f:
            labels.append(int(line))

    face_test = []
    for f in range(150):
        data = np.genfromtxt('./extracted_face_test_data/face_%d' % f, dtype=float)
        face_test.append(data)
    face_test = np.array(face_test)

    labels_test = []
    with open('./facedata/facedatatestlabels') as f:
        for line in f:
            labels_test.append(int(line))

    for epoch in range(5):
        w = [0 for i in range(70 * 60 + 1)]
        random_indexes = random.sample([i for i in range(451)],num_of_samples)
        start_time = default_timer()
        while (True):
            count = 0
            for i in range(len(random_indexes)):
                f = w[0]
                for row in range(70):
                    for col in range(60):
                        # print(faces.shape)
                        f += w[60*row+col] * (black_pixel_point if faces[random_indexes[i]][row][col] == 1.0 else white_pixel_point)
                if f < 0 and labels[random_indexes[i]] == 1:
                    count += 1
                    for j in range(1, 60*70):
                        w[j] = w[j] + (black_pixel_point if faces[random_indexes[i]][int((j-1) / 60)][(j-1) % 60] == 1.0 else white_pixel_point)
                    w[0] += 1
                if f >= 0 and labels[random_indexes[i]] == 0:
                    count += 1
                    for j in range(1, 60*70):
                        w[j] = w[j] - (black_pixel_point if faces[random_indexes[i]][int((j-1) / 60)][(j-1) % 60] == 1.0 else white_pixel_point)
                    w[0] -= 1

            end_time = default_timer()
            if count == 0 or end_time - start_time > looping_time:
                print(round(end_time-start_time,3),"s")
                break

        count = 0
        for i in range(len(labels_test)):
            f = w[0]
            for row in range(70):
                for col in range(60):
                    f += w[60*row+col] * (black_pixel_point if face_test[i][row][col] == 1.0 else white_pixel_point)
            if (f >= 0 and labels_test[i] == 1) or (f < 0 and labels_test[i] == 0):
                # print(f,labels_test[i])
                count += 1

        print(round(count / len(labels_test),3))
