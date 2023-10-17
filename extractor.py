import numpy as np

# # digit part
# total = []
# digit = []
# with open('./digitdata/trainingimages','r') as f:
#     for count,line in enumerate(f):
#         digit.append([line])
#         if((count+1) % 28 == 0):
#             total.append(digit)
#             digit = []
# np_arr = np.array(total)
#
# new_matrix = np.zeros((len(np_arr),28,28))
# for i in range(len(np_arr)):
#     for j in range(28):
#         for z in range(len(np_arr[i][j][0])):
#             if np_arr[i][j][0][z] == '#':
#                 new_matrix[i][j][z] = 1.0
#             elif np_arr[i][j][0][z] == '+':
#                 new_matrix[i][j][z] = 1.0
#     np.savetxt('./extracted_digit_training_data/digit_%d'%i,new_matrix[i],fmt='%s')

# face part

FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
total = []
face = []
with open('./facedata/facedatatest','r') as f:
    for count,line in enumerate(f):
        face.append([line])
        if((count+1) % 70 == 0):
            total.append(face)
            face = []
np_arr = np.array(total)

new_matrix = np.zeros((len(np_arr),70,60))
for i in range(len(np_arr)):
    for j in range(len(np_arr[i])):
        for z in range(len(np_arr[i][j][0])):
            if np_arr[i][j][0][z] == '#':
                new_matrix[i][j][z] = 1.0
            elif np_arr[i][j][0][z] == '+':
                new_matrix[i][j][z] = 1.0
    np.savetxt('./extracted_face_test_data/face_%d'%i,new_matrix[i],fmt='%s')