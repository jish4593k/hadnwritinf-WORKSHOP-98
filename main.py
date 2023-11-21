import os
import shutil
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

source_folder = "/media/ujwal/My files/Work/ML/Computer Vision/Handwriting Clustering/Input"
dest_folder = "/media/ujwal/My files/Work/ML/Computer Vision/Handwriting Clustering/Output"
cno = 10
mindim = 600


def load_images_from_folder(folder):
    data = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            data.update({filename: np.array(img[0:mindim, 0:mindim]).ravel()})
    return data


def remove_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def create_folder(directory):
    try:
        os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def k_means_clustering(data, cno):
    data_values = list(data.values())
    data_tensor = torch.FloatTensor(data_values)

    # Normalize the data
    data_tensor = (data_tensor - data_tensor.min()) / (data_tensor.max() - data_tensor.min())

    # Apply K-Means clustering using torch.kmeans
    centroids, labels = torch.kmeans(data_tensor.unsqueeze(0), cno)

    return labels.numpy()


def main():
    data = load_images_from_folder(source_folder)
    labels = k_means_clustering(data, cno)

    fol = []
    remove_folder(dest_folder)  # flush out previous outputs
    for i in range(cno):
        create_folder(dest_folder + "/Cluster " + str(i + 1))
        fol.append(dest_folder + "/Cluster " + str(i + 1))

    a = list(data.values())
    b = list(data.keys())
    for i in range(len(labels)):
        cv2.imwrite(fol[labels[i]] + "/" + b[i], a[i])


if __name__ == '__main__':
    main()
