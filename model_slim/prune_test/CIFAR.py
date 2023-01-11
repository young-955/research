import torchvision
torchvision.datasets.CIFAR10(root='../dataset/', download=True)
# %%
import pickle
import cv2
import numpy as np

def unpickle(file):    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

folder_path = '../dataset/cifar-10-batches-py/'
train_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_name = 'test_batch'

train_label = []
train_data = []
for n in train_names:
    data_batch = unpickle(f'{folder_path}{n}')
    labels = data_batch[b'labels']
    train_label += labels
    data = data_batch[b'data']
    for d in data:
        d = d.reshape(3, 32, 32)
        img = cv2.merge([d[0], d[1], d[2]])
        train_data.append(img)

with open('data/CIFAR10_train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('data/CIFAR10_train_label.pkl', 'wb') as f:
    pickle.dump(train_label, f)

test_label = []
test_data = []
data_batch = unpickle(f'{folder_path}{test_name}')
labels = data_batch[b'labels']
test_label += labels
data = data_batch[b'data']
for d in data:
    d = d.reshape(3, 32, 32)
    img = cv2.merge([d[0], d[1], d[2]])
    test_data.append(img)

with open('data/CIFAR10_test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

with open('data/CIFAR10_test_label.pkl', 'wb') as f:
    pickle.dump(test_label, f)


# %%
