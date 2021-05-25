from collections import defaultdict
from PIL import Image
from imutils import paths
import os
import pickle
import numpy as np

def prepare_caltech_data(root='/home/cytao/projects-summer/datasets'):
    # Get all files in the current directory
    image_paths = list(paths.list_images(os.path.join(root, '101_ObjectCategories')))
    labels = defaultdict(list)
    for i, image_path in enumerate(image_paths):
        label = image_path.split(os.path.sep)[-2]
        if label == 'BACKGROUND_Google':
            continue
        labels[label].append(i)


    train_index = []
    train_label= []
    test_index = []
    test_label = []
    for i, label in enumerate(labels):
        label_train_index =list(np.random.choice(labels[label],30))
        train_index.extend(label_train_index)
        train_label.extend([i]*len(label_train_index))
        label_test_index = list(set(labels[label])-set(label_train_index))
        test_index.extend(label_test_index)
        test_label.extend([i]*len(label_test_index))

    train_data=[]
    test_data=[]

    for i in train_index:
        train_data.append(Image.open(image_paths[i]).convert('RGB'))

    for i in test_index:
        test_data.append(Image.open(image_paths[i]).convert('RGB'))


    with open(os.path.join(root,  '101_ObjectCategories','train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
        pickle.dump(train_label, f)

    with open(os.path.join(root, '101_ObjectCategories', 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
        pickle.dump(test_label, f)

if __name__ == '__main__':
    prepare_caltech_data()
