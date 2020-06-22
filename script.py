import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
#print(digits)
print(digits.DESCR)
print(digits.data)
print(digits.target)

#Visualize the data images
plt.gray() 
#Image at index 100
plt.matshow(digits.images[100])
plt.show()
print(digits.target[100])#4

# Figure size (width, height)

fig = plt.figure(figsize=(6, 6))

# Adjust the subplots 

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images

for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position

    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position

    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value

    ax.text(0, 7, str(digits.target[i]))

plt.show()

#K-Means Clustering
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

#Visualizing after K-Means
fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
#The cluster centers should be a list with 64 values (0-16). Here, we are making each of the cluster centers into an 8x8 2D array.
plt.show()

#Testing the model
new_samples = np.array([
[0.00,0.00,0.00,1.67,3.04,0.00,0.00,0.00,0.00,0.30,5.47,7.47,7.62,0.00,0.00,0.00,0.00,1.51,7.62,5.93,7.62,1.06,0.00,0.00,0.00,0.00,0.53,1.90,7.62,2.28,0.00,0.00,0.00,0.00,5.01,7.62,7.62,6.63,2.04,0.00,0.00,0.38,7.47,7.62,7.62,7.23,7.62,0.91,0.00,0.60,7.62,7.62,5.62,0.68,2.13,0.00,0.00,0.00,1.90,1.90,0.00,0.00,0.00,0.00],
[0.00,0.00,1.90,4.50,5.02,1.29,0.00,0.00,0.00,5.24,7.62,7.62,7.62,7.17,1.28,0.00,0.00,7.62,5.10,1.90,3.56,7.55,5.79,0.00,0.00,6.85,4.87,0.00,0.00,5.48,6.09,0.00,0.00,4.87,7.39,3.11,4.04,7.62,4.95,0.00,0.00,0.60,6.47,7.62,7.32,5.86,0.68,0.00,0.00,0.00,0.07,1.98,0.46,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.68,2.59,3.73,4.57,1.59,0.00,0.00,0.00,6.24,7.62,7.54,7.54,4.49,0.00,0.00,0.00,1.44,1.52,0.30,7.24,4.11,0.00,0.00,0.00,0.00,0.00,1.37,7.62,2.66,0.00,0.00,0.00,0.00,0.00,2.36,7.62,1.75,0.00,0.00,0.00,0.00,0.00,1.67,5.94,0.61,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.44,2.82,1.29,0.30,0.00,0.00,0.00,0.00,4.10,7.62,7.62,5.17,0.00,0.00,0.00,0.00,3.03,7.62,7.62,7.08,0.30,0.00,0.00,0.00,0.00,3.27,7.62,7.62,5.25,0.00,0.00,0.00,0.00,0.00,3.26,7.62,7.62,2.35,0.00,0.00,0.00,0.00,4.18,7.62,7.62,7.08,0.30,0.00,0.00,0.00,3.57,6.78,6.86,4.56,0.08,0.00]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
print(new_labels)