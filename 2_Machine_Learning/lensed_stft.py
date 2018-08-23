import numpy as np

n_images = 5000

stft = np.load("/home/amitjit/output/spectrograms/PM/Lensed_1.npz")

lensed_stft = abs(stft['stft'].ravel())

for i in range(1, n_images):
    stft=np.load("/home/amitjit/output/spectrograms/PM/Lensed_" + str(i+1) + '.npz')
    print(i+1)
    lensed_stft =  np.vstack((lensed_stft, abs(stft['stft'].ravel())))

print(lensed_stft.shape)

np.savez("/home/amitjit/output/lensed_stft", lensed_stft=lensed_stft)
