import numpy as np
import imageio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import progressbar

progress = progressbar.ProgressBar()

#Number of samples of images of each class
n_samples = int(4000)


#Import image data as arrays
lensed_images = np.array([ imageio.imread('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/1. Lensed and Unlensed Model/Images/SIS_Lensed/SIS_Lensed_'+str(i+1)+'.png').ravel() for i in range(n_samples)])

lensed_labels = np.repeat("Lensed Wave", n_samples)
print('Finished reading lensed images.')

unlensed_images = np.array([ imageio.imread('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/1. Lensed and Unlensed Model/Images/Unlensed/Unlensed_'+str(i+1)+'.png').ravel() for i in range(n_samples)])

unlensed_labels = np.repeat("Unlensed Wave", n_samples)
print('Finished reading unlensed images.')

print('Standard scaling the data')
images = np.concatenate((lensed_images, unlensed_images))
scaler = StandardScaler()
images = scaler.fit_transform(images)
lensed_images, unlensed_images = images[:n_samples], images[n_samples:]

lensed_rank   = np.arange(n_samples) + 1
unlensed_rank = np.arange(n_samples) + 1


#Split train and test data
lensed_images_train, lensed_images_test, lensed_labels_train, lensed_labels_test, lensed_rank_train, lensed_rank_test = train_test_split(lensed_images,lensed_labels, lensed_rank)

unlensed = np.load('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/2. Machine Learning/data.npz')


print('Loading point mass data')
unlensed_images_train, unlensed_images_test, unlensed_labels_train, unlensed_labels_test, unlensed_rank_test = unlensed[unlensed.files[0]], unlensed[unlensed.files[1]], unlensed[unlensed.files[2]], unlensed[unlensed.files[3]], unlensed[unlensed.files[4]]


print('Filtering 1')

shift1 = 0
for i, name in progress(enumerate(unlensed_labels_train)):
    print(name)
    if name=='Lensed Wave':
        unlensed_labels_train = np.delete(unlensed_labels_train, i-shift1)
        unlensed_images_train = np.delete(unlensed_images_train, i-shift1)
        shift1 += 1
        
        
print('Filtering 2')
shift2 = 0
for i, name in progress(enumerate(unlensed_labels_test)):
    print(name)
    if name=='Lensed Wave':
        unlensed_labels_test = np.delete(unlensed_labels_test, i-shift2)
        unlensed_images_test = np.delete(unlensed_images_test, i-shift2)
        unlensed_rank_test   = np.delete(unlensed_rank_test  , i-shift2)
        shift2 += 1

print('Filtering is done, Concatenating')

X_train = np.concatenate((lensed_images_train, unlensed_images_train))
X_test  = np.concatenate((lensed_images_test, unlensed_images_test))

y_train = np.concatenate((lensed_labels_train, unlensed_labels_train))
y_test  = np.concatenate((lensed_labels_test, unlensed_labels_test))

rank_test = np.concatenate((lensed_rank_test, unlensed_rank_test))


np.savez('C:/Users/Amitjit/OneDrive - The Chinese University of Hong Kong/Internship (Summer 2018)/Code/2. Machine Learning/data_SIS', X_train = X_train, X_test = X_test, y_test = y_test, y_train = y_train, rank_test = rank_test)


#Completion sound
import winsound
duration = 900  # millisecond
freq = 850  # Hz
winsound.Beep(freq, duration)