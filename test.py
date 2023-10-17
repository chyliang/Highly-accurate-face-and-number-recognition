import numpy as np
import operator
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import random
from timeit import default_timer

def euc_dist(x1, x2):
     return np.sqrt(np.sum((x1-x2)**2))
class KNN:
     def __init__(self, K=3):
          self.K = K
     def fit(self, x_train, y_train):
          self.X_train = x_train
          self.Y_train = y_train
     def predict(self, X_test):
          predictions = []
          for i in range(len(X_test)):
               dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train])
               dist_sorted = dist.argsort()[:self.K]
               neigh_count = {}
               for idx in dist_sorted:
                    if self.Y_train[idx] in neigh_count:
                         neigh_count[self.Y_train[idx]] += 1
                    else:
                         neigh_count[self.Y_train[idx]] = 1
               sorted_neigh_count = sorted(neigh_count.items(), key=operator.itemgetter(1), reverse=True)
               predictions.append(sorted_neigh_count[0][0])
          return predictions

y_train = []
with open('./digitdata/traininglabels') as f:
     for line in f:
          y_train.append(int(line))

directory = './extracted_digit_training_data'

x_train = []
# get x, and construct training set
for f in range(5000):
     data = np.genfromtxt(directory + '/digit_%d' % f, dtype=float)
     x_train.append(data.reshape(784))

y_test = []
with open('./digitdata/testlabels') as f:
     for line in f:
          y_test.append(int(line))

x_test = []
for f in range(1000):
     data_test = np.genfromtxt("./extracted_digit_test_data/digit_%d" % f, dtype=float)
     x_test.append(data_test.reshape(784))

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

random_indexes = random.sample([i for i in range(5000)],int(5000*0.6))

kVals = np.arange(3,100,2)
accuracies = []
start_time = default_timer()
for k in kVals:
     model = KNN(K = k)
     model.fit(x_train[random_indexes], y_train[random_indexes])
     pred = model.predict(x_test)
     acc = accuracy_score(y_test, pred)
     accuracies.append(acc)
     print("K = "+str(k)+"; Accuracy: "+str(acc))
end_time = default_timer()
print(round(end_time-start_time,3),"s")