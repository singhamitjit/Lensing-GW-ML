import numpy as np
import imageio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

#Number of samples of images of each class
n_samples = int(5000)


#Import image data as arrays
lensed_images = np.array([ imageio.imread('/home/amitjit/output/spectrograms/PM/Lensed_' +str(i+1) + '.png').ravel() for i in range(n_samples)])

lensed_labels = np.repeat("Lensed Wave", n_samples)
print('Finished reading lensed images.')

unlensed_images = np.array([ imageio.imread('/home/amitjit/output/spectrograms/Unlensed/Unlensed_' +str(i+1) + '.png').ravel() for i in range(n_samples)])

unlensed_labels = np.repeat("Unlensed Wave", n_samples)
print('Finished reading unlensed images.')

print('Standard scaling the data')
images = np.concatenate((lensed_images, unlensed_images))
scaler = StandardScaler()
images = scaler.fit_transform(images)

lensed_images, unlensed_images = None, None

lensed_images, unlensed_images = images[:n_samples], images[n_samples:]



lensed_rank   = np.arange(n_samples) + 1
unlensed_rank = np.arange(n_samples) + 1


#Split train and test data
lensed_images_train, lensed_images_test, lensed_labels_train, lensed_labels_test, lensed_rank_train, lensed_rank_test = train_test_split(lensed_images,lensed_labels, lensed_rank)

unlensed_images_train, unlensed_images_test, unlensed_labels_train, unlensed_labels_test, unlensed_rank_train, unlensed_rank_test = train_test_split(unlensed_images, unlensed_labels, unlensed_rank)

X_train = np.concatenate((lensed_images_train, unlensed_images_train))
X_test  = np.concatenate((lensed_images_test, unlensed_images_test))

y_train = np.concatenate((lensed_labels_train, unlensed_labels_train))
y_test  = np.concatenate((lensed_labels_test, unlensed_labels_test))

rank_test = np.concatenate((lensed_rank_test, unlensed_rank_test))


np.savez('/home/amitjit/output/data_PM', X_train = X_train, X_test = X_test, y_test = y_test, y_train = y_train, rank_test = rank_test)

