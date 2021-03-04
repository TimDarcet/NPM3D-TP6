NPM3D - TP6
===========

## Question 1

Here are the accuracies obtained with the pointMLP.

| epoch | accuracy on ModelNet10_PLY | accuracy on ModelNet40_PLY|
|-------|----------------------------|---------------------------|
| 25 | 19.4 % | 19.4 % |
| 250 | 28.5 % | 22.9 % |

Although these are not high accuracies, they are still significantly better than random accuracy (which would be 10% on 10 classes and 2.5% and 40 classes). Unsurprisingly, the accuracy is better where there are less classes.


## Question 2

Here are the accuracies obtained with the basic PointNet, without the Tnet.

epoch | accuracy on ModelNet10_PLY | accuracy on ModelNet40_PLY
25 | 88.0 % | 81.7 %

These accuracies are incredibly good compared to the previous ones, especially since these were obtained very easily, without any tuning and without needing much work to code the architecture. Again, the accuracy is better on the 10 classes dataset than on the 40 classes dataset, which was expected.




----------------------

MLP
train_ds = PointCloudData("../data/ModelNet40_PLY")
test_ds = PointCloudData("../data/ModelNet40_PLY", folder='test')

Classes:  {0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf', 5: 'bottle', 6: 'bowl', 7: 'car', 8: 'chair', 9: 'cone', 10: 'cup', 11: 'curtain', 12: 'desk', 13: 'door', 14: 'dresser', 15: 'flower_pot', 16: 'glass_box', 17: 'guitar', 18: 'keyboard', 19: 'lamp', 20: 'laptop', 21: 'mantel', 22: 'monitor', 23: 'night_stand', 24: 'person', 25: 'piano', 26: 'plant', 27: 'radio', 28: 'range_hood', 29: 'sink', 30: 'sofa', 31: 'stairs', 32: 'stool', 33: 'table', 34: 'tent', 35: 'toilet', 36: 'tv_stand', 37: 'vase', 38: 'wardrobe', 39: 'xbox'}
Train dataset size:  9843
Test dataset size:  2468
Number of classes:  40
Sample pointcloud shape:  torch.Size([1024, 3])
Number of parameters in the Neural Networks:  1716600
Device:  cuda:0
Epoch: 1, Loss: 3.884, Test accuracy: 6.0 %
Epoch: 2, Loss: 3.533, Test accuracy: 7.0 %
Epoch: 3, Loss: 3.645, Test accuracy: 7.5 %
Epoch: 4, Loss: 3.339, Test accuracy: 7.7 %
Epoch: 5, Loss: 3.310, Test accuracy: 8.5 %
Epoch: 6, Loss: 3.375, Test accuracy: 10.0 %
Epoch: 7, Loss: 3.523, Test accuracy: 10.5 %
Epoch: 8, Loss: 3.126, Test accuracy: 11.3 %
Epoch: 9, Loss: 3.233, Test accuracy: 14.5 %
Epoch: 10, Loss: 2.993, Test accuracy: 14.3 %
Epoch: 11, Loss: 3.564, Test accuracy: 14.2 %
Epoch: 12, Loss: 3.027, Test accuracy: 14.7 %
Epoch: 13, Loss: 3.090, Test accuracy: 15.9 %
Epoch: 14, Loss: 3.101, Test accuracy: 17.0 %
Epoch: 15, Loss: 2.926, Test accuracy: 15.2 %
Epoch: 16, Loss: 2.984, Test accuracy: 15.7 %
Epoch: 17, Loss: 3.270, Test accuracy: 18.1 %
Epoch: 18, Loss: 3.034, Test accuracy: 16.9 %
Epoch: 19, Loss: 2.288, Test accuracy: 17.1 %
Epoch: 20, Loss: 3.022, Test accuracy: 17.9 %
Epoch: 21, Loss: 2.985, Test accuracy: 18.9 %
Epoch: 22, Loss: 3.112, Test accuracy: 20.0 %
Epoch: 23, Loss: 3.023, Test accuracy: 19.1 %
Epoch: 24, Loss: 2.665, Test accuracy: 19.7 %
Epoch: 25, Loss: 2.784, Test accuracy: 19.4 %
Epoch: 26, Loss: 2.908, Test accuracy: 20.1 %
Epoch: 27, Loss: 2.863, Test accuracy: 19.6 %
Epoch: 28, Loss: 2.473, Test accuracy: 20.0 %
Epoch: 29, Loss: 2.541, Test accuracy: 19.2 %
Epoch: 30, Loss: 2.320, Test accuracy: 20.5 %
Epoch: 31, Loss: 2.493, Test accuracy: 20.4 %
Epoch: 32, Loss: 2.815, Test accuracy: 20.8 %
Epoch: 33, Loss: 2.557, Test accuracy: 19.4 %
Epoch: 34, Loss: 2.838, Test accuracy: 20.7 %
Epoch: 35, Loss: 2.863, Test accuracy: 21.2 %
Epoch: 36, Loss: 2.561, Test accuracy: 20.6 %
Epoch: 37, Loss: 2.410, Test accuracy: 19.8 %
Epoch: 38, Loss: 2.858, Test accuracy: 21.3 %
Epoch: 39, Loss: 2.812, Test accuracy: 21.2 %
Epoch: 40, Loss: 2.981, Test accuracy: 20.5 %
Epoch: 41, Loss: 2.710, Test accuracy: 21.2 %
Epoch: 42, Loss: 2.775, Test accuracy: 20.1 %
Epoch: 43, Loss: 2.422, Test accuracy: 20.7 %
Epoch: 44, Loss: 2.959, Test accuracy: 19.9 %
Epoch: 45, Loss: 2.776, Test accuracy: 20.8 %
Epoch: 46, Loss: 2.700, Test accuracy: 21.3 %
Epoch: 47, Loss: 2.529, Test accuracy: 21.8 %
Epoch: 48, Loss: 2.416, Test accuracy: 21.2 %
Epoch: 49, Loss: 2.467, Test accuracy: 22.1 %
Epoch: 50, Loss: 2.280, Test accuracy: 21.0 %
Epoch: 51, Loss: 2.359, Test accuracy: 21.4 %
Epoch: 52, Loss: 2.445, Test accuracy: 22.1 %
Epoch: 53, Loss: 2.292, Test accuracy: 21.8 %
Epoch: 54, Loss: 2.483, Test accuracy: 22.4 %
Epoch: 55, Loss: 2.812, Test accuracy: 21.7 %
Epoch: 56, Loss: 2.778, Test accuracy: 20.9 %
Epoch: 57, Loss: 2.673, Test accuracy: 21.2 %
Epoch: 58, Loss: 2.879, Test accuracy: 21.4 %
Epoch: 59, Loss: 2.752, Test accuracy: 21.2 %
Epoch: 60, Loss: 2.561, Test accuracy: 23.0 %
Epoch: 61, Loss: 2.800, Test accuracy: 22.4 %
Epoch: 62, Loss: 2.773, Test accuracy: 22.2 %
Epoch: 63, Loss: 2.612, Test accuracy: 22.2 %
Epoch: 64, Loss: 2.635, Test accuracy: 22.4 %
Epoch: 65, Loss: 2.726, Test accuracy: 22.9 %
Epoch: 66, Loss: 2.629, Test accuracy: 22.2 %
Epoch: 67, Loss: 2.758, Test accuracy: 21.8 %
Epoch: 68, Loss: 2.253, Test accuracy: 21.9 %
Epoch: 69, Loss: 2.681, Test accuracy: 21.6 %
Epoch: 70, Loss: 2.588, Test accuracy: 22.9 %
Epoch: 71, Loss: 2.979, Test accuracy: 22.9 %
Epoch: 72, Loss: 2.663, Test accuracy: 22.4 %
Epoch: 73, Loss: 2.567, Test accuracy: 21.6 %
Epoch: 74, Loss: 2.540, Test accuracy: 22.1 %
Epoch: 75, Loss: 2.993, Test accuracy: 22.6 %
Epoch: 76, Loss: 2.782, Test accuracy: 23.3 %
Epoch: 77, Loss: 2.497, Test accuracy: 21.5 %
Epoch: 78, Loss: 2.609, Test accuracy: 21.9 %
Epoch: 79, Loss: 2.623, Test accuracy: 22.6 %
Epoch: 80, Loss: 2.690, Test accuracy: 22.1 %
Epoch: 81, Loss: 2.826, Test accuracy: 22.9 %
Epoch: 82, Loss: 2.718, Test accuracy: 23.2 %
Epoch: 83, Loss: 2.601, Test accuracy: 22.7 %
Epoch: 84, Loss: 2.743, Test accuracy: 22.9 %
Epoch: 85, Loss: 3.032, Test accuracy: 22.6 %
Epoch: 86, Loss: 2.478, Test accuracy: 23.4 %
Epoch: 87, Loss: 2.650, Test accuracy: 21.9 %
Epoch: 88, Loss: 2.573, Test accuracy: 23.1 %
Epoch: 89, Loss: 2.553, Test accuracy: 21.6 %
Epoch: 90, Loss: 2.829, Test accuracy: 23.1 %
Epoch: 91, Loss: 2.352, Test accuracy: 22.3 %
Epoch: 92, Loss: 2.668, Test accuracy: 22.9 %
Epoch: 93, Loss: 2.827, Test accuracy: 22.4 %
Epoch: 94, Loss: 2.322, Test accuracy: 22.2 %
Epoch: 95, Loss: 2.779, Test accuracy: 22.3 %
Epoch: 96, Loss: 2.701, Test accuracy: 23.2 %
Epoch: 97, Loss: 2.597, Test accuracy: 22.6 %
Epoch: 98, Loss: 3.062, Test accuracy: 22.0 %
Epoch: 99, Loss: 2.166, Test accuracy: 22.4 %
Epoch: 100, Loss: 2.650, Test accuracy: 22.3 %
Epoch: 101, Loss: 2.613, Test accuracy: 22.4 %
Epoch: 102, Loss: 2.701, Test accuracy: 22.3 %
Epoch: 103, Loss: 1.982, Test accuracy: 23.1 %
Epoch: 104, Loss: 3.024, Test accuracy: 22.6 %
Epoch: 105, Loss: 2.204, Test accuracy: 23.6 %
Epoch: 106, Loss: 2.673, Test accuracy: 23.2 %
Epoch: 107, Loss: 2.737, Test accuracy: 22.8 %
Epoch: 108, Loss: 2.616, Test accuracy: 23.1 %
Epoch: 109, Loss: 2.631, Test accuracy: 22.8 %
Epoch: 110, Loss: 2.450, Test accuracy: 23.0 %
Epoch: 111, Loss: 2.338, Test accuracy: 23.3 %
Epoch: 112, Loss: 3.201, Test accuracy: 22.9 %
Epoch: 113, Loss: 2.940, Test accuracy: 22.5 %
Epoch: 114, Loss: 3.118, Test accuracy: 22.7 %
Epoch: 115, Loss: 2.545, Test accuracy: 22.7 %
Epoch: 116, Loss: 2.586, Test accuracy: 22.7 %
Epoch: 117, Loss: 2.277, Test accuracy: 22.8 %
Epoch: 118, Loss: 2.131, Test accuracy: 22.6 %
Epoch: 119, Loss: 3.007, Test accuracy: 22.3 %
Epoch: 120, Loss: 2.826, Test accuracy: 24.0 %
Epoch: 121, Loss: 2.616, Test accuracy: 23.3 %
Epoch: 122, Loss: 2.088, Test accuracy: 23.4 %
Epoch: 123, Loss: 2.405, Test accuracy: 22.5 %
Epoch: 124, Loss: 2.321, Test accuracy: 22.6 %
Epoch: 125, Loss: 2.236, Test accuracy: 22.9 %
Epoch: 126, Loss: 2.508, Test accuracy: 23.2 %
Epoch: 127, Loss: 2.407, Test accuracy: 22.9 %
Epoch: 128, Loss: 2.511, Test accuracy: 23.9 %
Epoch: 129, Loss: 2.238, Test accuracy: 22.5 %
Epoch: 130, Loss: 2.597, Test accuracy: 24.0 %
Epoch: 131, Loss: 2.383, Test accuracy: 23.3 %
Epoch: 132, Loss: 2.595, Test accuracy: 22.9 %
Epoch: 133, Loss: 2.376, Test accuracy: 22.2 %
Epoch: 134, Loss: 2.922, Test accuracy: 23.9 %
Epoch: 135, Loss: 2.456, Test accuracy: 23.5 %
Epoch: 136, Loss: 2.762, Test accuracy: 23.0 %
Epoch: 137, Loss: 2.373, Test accuracy: 22.9 %
Epoch: 138, Loss: 3.086, Test accuracy: 23.1 %
Epoch: 139, Loss: 2.499, Test accuracy: 24.4 %
Epoch: 140, Loss: 2.577, Test accuracy: 22.9 %
Epoch: 141, Loss: 2.426, Test accuracy: 22.4 %
Epoch: 142, Loss: 3.137, Test accuracy: 23.7 %
Epoch: 143, Loss: 2.253, Test accuracy: 23.3 %
Epoch: 144, Loss: 2.787, Test accuracy: 23.5 %
Epoch: 145, Loss: 2.208, Test accuracy: 23.1 %
Epoch: 146, Loss: 2.511, Test accuracy: 22.5 %
Epoch: 147, Loss: 2.109, Test accuracy: 22.3 %
Epoch: 148, Loss: 2.817, Test accuracy: 23.1 %
Epoch: 149, Loss: 2.196, Test accuracy: 22.6 %
Epoch: 150, Loss: 2.359, Test accuracy: 23.1 %
Epoch: 151, Loss: 2.825, Test accuracy: 22.6 %
Epoch: 152, Loss: 3.044, Test accuracy: 23.6 %
Epoch: 153, Loss: 2.317, Test accuracy: 22.6 %
Epoch: 154, Loss: 2.721, Test accuracy: 22.8 %
Epoch: 155, Loss: 2.611, Test accuracy: 23.3 %
Epoch: 156, Loss: 2.564, Test accuracy: 23.4 %
Epoch: 157, Loss: 2.339, Test accuracy: 23.9 %
Epoch: 158, Loss: 2.181, Test accuracy: 22.2 %
Epoch: 159, Loss: 2.472, Test accuracy: 24.0 %
Epoch: 160, Loss: 2.518, Test accuracy: 23.9 %
Epoch: 161, Loss: 2.273, Test accuracy: 23.1 %
Epoch: 162, Loss: 2.419, Test accuracy: 22.4 %
Epoch: 163, Loss: 2.713, Test accuracy: 23.1 %
Epoch: 164, Loss: 2.570, Test accuracy: 24.1 %
Epoch: 165, Loss: 2.575, Test accuracy: 22.3 %
Epoch: 166, Loss: 2.740, Test accuracy: 22.9 %
Epoch: 167, Loss: 2.474, Test accuracy: 22.6 %
Epoch: 168, Loss: 2.712, Test accuracy: 22.6 %
Epoch: 169, Loss: 2.729, Test accuracy: 23.5 %
Epoch: 170, Loss: 2.763, Test accuracy: 22.9 %
Epoch: 171, Loss: 2.882, Test accuracy: 23.6 %
Epoch: 172, Loss: 2.241, Test accuracy: 23.9 %
Epoch: 173, Loss: 2.942, Test accuracy: 23.1 %
Epoch: 174, Loss: 2.828, Test accuracy: 24.1 %
Epoch: 175, Loss: 2.384, Test accuracy: 23.6 %
Epoch: 176, Loss: 2.643, Test accuracy: 23.1 %
Epoch: 177, Loss: 3.010, Test accuracy: 24.2 %
Epoch: 178, Loss: 2.435, Test accuracy: 23.7 %
Epoch: 179, Loss: 2.688, Test accuracy: 22.7 %
Epoch: 180, Loss: 2.773, Test accuracy: 22.5 %
Epoch: 181, Loss: 2.389, Test accuracy: 23.9 %
Epoch: 182, Loss: 2.554, Test accuracy: 23.8 %
Epoch: 183, Loss: 2.295, Test accuracy: 23.8 %
Epoch: 184, Loss: 2.625, Test accuracy: 23.1 %
Epoch: 185, Loss: 2.342, Test accuracy: 23.0 %
Epoch: 186, Loss: 2.730, Test accuracy: 23.0 %
Epoch: 187, Loss: 2.652, Test accuracy: 23.3 %
Epoch: 188, Loss: 2.644, Test accuracy: 23.2 %
Epoch: 189, Loss: 2.438, Test accuracy: 22.8 %
Epoch: 190, Loss: 2.534, Test accuracy: 23.8 %
Epoch: 191, Loss: 2.559, Test accuracy: 23.3 %
Epoch: 192, Loss: 2.626, Test accuracy: 23.3 %
Epoch: 193, Loss: 2.611, Test accuracy: 23.1 %
Epoch: 194, Loss: 2.880, Test accuracy: 23.3 %
Epoch: 195, Loss: 2.947, Test accuracy: 23.6 %
Epoch: 196, Loss: 2.734, Test accuracy: 23.4 %
Epoch: 197, Loss: 2.356, Test accuracy: 23.1 %
Epoch: 198, Loss: 2.727, Test accuracy: 23.8 %
Epoch: 199, Loss: 2.559, Test accuracy: 22.2 %
Epoch: 200, Loss: 2.678, Test accuracy: 23.4 %
Epoch: 201, Loss: 2.119, Test accuracy: 23.1 %
Epoch: 202, Loss: 2.734, Test accuracy: 22.7 %
Epoch: 203, Loss: 2.310, Test accuracy: 22.8 %
Epoch: 204, Loss: 2.284, Test accuracy: 23.5 %
Epoch: 205, Loss: 2.319, Test accuracy: 23.4 %
Epoch: 206, Loss: 3.059, Test accuracy: 22.8 %
Epoch: 207, Loss: 2.680, Test accuracy: 22.9 %
Epoch: 208, Loss: 2.689, Test accuracy: 22.9 %
Epoch: 209, Loss: 2.918, Test accuracy: 23.0 %
Epoch: 210, Loss: 3.072, Test accuracy: 23.9 %
Epoch: 211, Loss: 2.457, Test accuracy: 24.1 %
Epoch: 212, Loss: 2.600, Test accuracy: 24.6 %
Epoch: 213, Loss: 2.749, Test accuracy: 23.4 %
Epoch: 214, Loss: 2.925, Test accuracy: 23.1 %
Epoch: 215, Loss: 2.823, Test accuracy: 23.3 %
Epoch: 216, Loss: 2.521, Test accuracy: 23.7 %
Epoch: 217, Loss: 2.538, Test accuracy: 22.5 %
Epoch: 218, Loss: 2.562, Test accuracy: 22.3 %
Epoch: 219, Loss: 2.516, Test accuracy: 23.5 %
Epoch: 220, Loss: 2.785, Test accuracy: 22.4 %
Epoch: 221, Loss: 2.682, Test accuracy: 22.9 %
Epoch: 222, Loss: 2.853, Test accuracy: 23.3 %
Epoch: 223, Loss: 2.563, Test accuracy: 23.3 %
Epoch: 224, Loss: 2.207, Test accuracy: 24.1 %
Epoch: 225, Loss: 2.527, Test accuracy: 24.2 %
Epoch: 226, Loss: 2.650, Test accuracy: 23.8 %
Epoch: 227, Loss: 2.779, Test accuracy: 22.9 %
Epoch: 228, Loss: 2.228, Test accuracy: 23.2 %
Epoch: 229, Loss: 2.241, Test accuracy: 23.7 %
Epoch: 230, Loss: 2.757, Test accuracy: 22.0 %
Epoch: 231, Loss: 2.325, Test accuracy: 24.1 %
Epoch: 232, Loss: 2.825, Test accuracy: 22.6 %
Epoch: 233, Loss: 2.444, Test accuracy: 23.1 %
Epoch: 234, Loss: 2.547, Test accuracy: 23.9 %
Epoch: 235, Loss: 2.447, Test accuracy: 23.2 %
Epoch: 236, Loss: 2.492, Test accuracy: 23.7 %
Epoch: 237, Loss: 2.646, Test accuracy: 23.3 %
Epoch: 238, Loss: 2.372, Test accuracy: 23.2 %
Epoch: 239, Loss: 2.494, Test accuracy: 23.6 %
Epoch: 240, Loss: 2.330, Test accuracy: 22.7 %
Epoch: 241, Loss: 2.856, Test accuracy: 23.4 %
Epoch: 242, Loss: 2.987, Test accuracy: 23.4 %
Epoch: 243, Loss: 2.627, Test accuracy: 23.3 %
Epoch: 244, Loss: 2.696, Test accuracy: 23.8 %
Epoch: 245, Loss: 2.757, Test accuracy: 23.3 %
Epoch: 246, Loss: 2.525, Test accuracy: 24.0 %
Epoch: 247, Loss: 2.261, Test accuracy: 22.4 %
Epoch: 248, Loss: 3.382, Test accuracy: 23.8 %
Epoch: 249, Loss: 2.873, Test accuracy: 22.3 %
Epoch: 250, Loss: 2.634, Test accuracy: 22.9 %
Total time for training :  3659.1826202869415



MLP
train_ds = PointCloudData("../data/ModelNet10_PLY")
test_ds = PointCloudData("../data/ModelNet10_PLY", folder='test')

Classes:  {0: 'bathtub', 1: 'bed', 2: 'chair', 3: 'desk', 4: 'dresser', 5: 'monitor', 6: 'night_stand', 7: 'sofa', 8: 'table', 9: 'toilet'}
Train dataset size:  3991
Test dataset size:  908
Number of classes:  10
Sample pointcloud shape:  torch.Size([1024, 3])
Number of parameters in the Neural Networks:  1716600
Device:  cuda:0
Epoch: 1, Loss: 3.538, Test accuracy: 12.6 %
Epoch: 2, Loss: 3.406, Test accuracy: 14.6 %
Epoch: 3, Loss: 3.206, Test accuracy: 14.3 %
Epoch: 4, Loss: 3.002, Test accuracy: 14.4 %
Epoch: 5, Loss: 2.663, Test accuracy: 13.7 %
Epoch: 6, Loss: 2.880, Test accuracy: 15.0 %
Epoch: 7, Loss: 3.022, Test accuracy: 13.9 %
Epoch: 8, Loss: 2.776, Test accuracy: 13.4 %
Epoch: 9, Loss: 2.618, Test accuracy: 16.0 %
Epoch: 10, Loss: 2.644, Test accuracy: 15.9 %
Epoch: 11, Loss: 2.454, Test accuracy: 15.9 %
Epoch: 12, Loss: 2.130, Test accuracy: 15.4 %
Epoch: 13, Loss: 2.327, Test accuracy: 15.6 %
Epoch: 14, Loss: 2.533, Test accuracy: 18.5 %
Epoch: 15, Loss: 2.070, Test accuracy: 16.7 %
Epoch: 16, Loss: 2.107, Test accuracy: 17.7 %
Epoch: 17, Loss: 2.236, Test accuracy: 17.1 %
Epoch: 18, Loss: 2.323, Test accuracy: 16.6 %
Epoch: 19, Loss: 2.116, Test accuracy: 17.6 %
Epoch: 20, Loss: 2.217, Test accuracy: 16.6 %
Epoch: 21, Loss: 2.203, Test accuracy: 17.2 %
Epoch: 22, Loss: 2.140, Test accuracy: 18.7 %
Epoch: 23, Loss: 1.890, Test accuracy: 20.5 %
Epoch: 24, Loss: 2.225, Test accuracy: 18.9 %
Epoch: 25, Loss: 1.995, Test accuracy: 19.4 %
Epoch: 26, Loss: 2.172, Test accuracy: 20.9 %
Epoch: 27, Loss: 2.010, Test accuracy: 21.6 %
Epoch: 28, Loss: 2.034, Test accuracy: 20.9 %
Epoch: 29, Loss: 2.200, Test accuracy: 20.5 %
Epoch: 30, Loss: 1.872, Test accuracy: 19.3 %
Epoch: 31, Loss: 1.815, Test accuracy: 20.5 %
Epoch: 32, Loss: 2.544, Test accuracy: 19.8 %
Epoch: 33, Loss: 2.032, Test accuracy: 21.1 %
Epoch: 34, Loss: 1.974, Test accuracy: 21.8 %
Epoch: 35, Loss: 1.803, Test accuracy: 20.9 %
Epoch: 36, Loss: 1.970, Test accuracy: 22.6 %
Epoch: 37, Loss: 2.041, Test accuracy: 21.6 %
Epoch: 38, Loss: 2.089, Test accuracy: 23.6 %
Epoch: 39, Loss: 1.840, Test accuracy: 21.6 %
Epoch: 40, Loss: 1.928, Test accuracy: 22.9 %
Epoch: 41, Loss: 1.791, Test accuracy: 23.2 %
Epoch: 42, Loss: 1.686, Test accuracy: 23.3 %
Epoch: 43, Loss: 1.739, Test accuracy: 23.9 %
Epoch: 44, Loss: 1.864, Test accuracy: 21.5 %
Epoch: 45, Loss: 1.920, Test accuracy: 22.1 %
Epoch: 46, Loss: 2.211, Test accuracy: 23.2 %
Epoch: 47, Loss: 1.682, Test accuracy: 23.7 %
Epoch: 48, Loss: 2.069, Test accuracy: 23.3 %
Epoch: 49, Loss: 1.848, Test accuracy: 25.0 %
Epoch: 50, Loss: 1.542, Test accuracy: 23.1 %
Epoch: 51, Loss: 1.756, Test accuracy: 25.4 %
Epoch: 52, Loss: 1.590, Test accuracy: 24.4 %
Epoch: 53, Loss: 1.791, Test accuracy: 24.2 %
Epoch: 54, Loss: 1.640, Test accuracy: 24.7 %
Epoch: 55, Loss: 1.751, Test accuracy: 24.0 %
Epoch: 56, Loss: 1.928, Test accuracy: 24.8 %
Epoch: 57, Loss: 1.933, Test accuracy: 25.8 %
Epoch: 58, Loss: 1.576, Test accuracy: 25.3 %
Epoch: 59, Loss: 1.459, Test accuracy: 24.6 %
Epoch: 60, Loss: 1.612, Test accuracy: 25.0 %
Epoch: 61, Loss: 1.593, Test accuracy: 24.8 %
Epoch: 62, Loss: 1.999, Test accuracy: 25.2 %
Epoch: 63, Loss: 1.971, Test accuracy: 24.9 %
Epoch: 64, Loss: 1.551, Test accuracy: 24.7 %
Epoch: 65, Loss: 1.405, Test accuracy: 24.7 %
Epoch: 66, Loss: 1.441, Test accuracy: 25.0 %
Epoch: 67, Loss: 1.601, Test accuracy: 26.1 %
Epoch: 68, Loss: 1.445, Test accuracy: 25.1 %
Epoch: 69, Loss: 1.807, Test accuracy: 25.7 %
Epoch: 70, Loss: 1.867, Test accuracy: 26.2 %
Epoch: 71, Loss: 1.828, Test accuracy: 26.1 %
Epoch: 72, Loss: 1.560, Test accuracy: 25.7 %
Epoch: 73, Loss: 1.937, Test accuracy: 27.2 %
Epoch: 74, Loss: 1.770, Test accuracy: 25.6 %
Epoch: 75, Loss: 1.550, Test accuracy: 29.0 %
Epoch: 76, Loss: 1.635, Test accuracy: 26.1 %
Epoch: 77, Loss: 1.730, Test accuracy: 25.7 %
Epoch: 78, Loss: 1.730, Test accuracy: 26.7 %
Epoch: 79, Loss: 1.613, Test accuracy: 25.7 %
Epoch: 80, Loss: 1.517, Test accuracy: 26.4 %
Epoch: 81, Loss: 2.057, Test accuracy: 26.5 %
Epoch: 82, Loss: 2.014, Test accuracy: 24.8 %
Epoch: 83, Loss: 1.772, Test accuracy: 25.9 %
Epoch: 84, Loss: 1.351, Test accuracy: 26.5 %
Epoch: 85, Loss: 1.998, Test accuracy: 26.2 %
Epoch: 86, Loss: 1.606, Test accuracy: 26.9 %
Epoch: 87, Loss: 1.618, Test accuracy: 26.4 %
Epoch: 88, Loss: 1.984, Test accuracy: 28.7 %
Epoch: 89, Loss: 1.716, Test accuracy: 25.8 %
Epoch: 90, Loss: 1.748, Test accuracy: 27.3 %
Epoch: 91, Loss: 1.524, Test accuracy: 27.9 %
Epoch: 92, Loss: 1.302, Test accuracy: 28.1 %
Epoch: 93, Loss: 1.646, Test accuracy: 27.5 %
Epoch: 94, Loss: 1.625, Test accuracy: 26.1 %
Epoch: 95, Loss: 1.668, Test accuracy: 26.1 %
Epoch: 96, Loss: 1.617, Test accuracy: 26.5 %
Epoch: 97, Loss: 1.719, Test accuracy: 27.3 %
Epoch: 98, Loss: 1.703, Test accuracy: 27.0 %
Epoch: 99, Loss: 1.585, Test accuracy: 25.7 %
Epoch: 100, Loss: 1.905, Test accuracy: 27.8 %
Epoch: 101, Loss: 1.392, Test accuracy: 27.8 %
Epoch: 102, Loss: 1.567, Test accuracy: 26.4 %
Epoch: 103, Loss: 1.299, Test accuracy: 26.7 %
Epoch: 104, Loss: 2.114, Test accuracy: 28.7 %
Epoch: 105, Loss: 1.945, Test accuracy: 27.0 %
Epoch: 106, Loss: 1.654, Test accuracy: 27.5 %
Epoch: 107, Loss: 1.647, Test accuracy: 26.8 %
Epoch: 108, Loss: 2.116, Test accuracy: 26.4 %
Epoch: 109, Loss: 1.758, Test accuracy: 27.2 %
Epoch: 110, Loss: 1.502, Test accuracy: 26.5 %
Epoch: 111, Loss: 1.419, Test accuracy: 26.8 %
Epoch: 112, Loss: 1.859, Test accuracy: 27.2 %
Epoch: 113, Loss: 1.635, Test accuracy: 26.1 %
Epoch: 114, Loss: 1.753, Test accuracy: 26.3 %
Epoch: 115, Loss: 1.404, Test accuracy: 27.8 %
Epoch: 116, Loss: 1.793, Test accuracy: 27.0 %
Epoch: 117, Loss: 1.433, Test accuracy: 28.3 %
Epoch: 118, Loss: 1.376, Test accuracy: 27.2 %
Epoch: 119, Loss: 1.862, Test accuracy: 28.0 %
Epoch: 120, Loss: 1.517, Test accuracy: 27.2 %
Epoch: 121, Loss: 1.512, Test accuracy: 28.0 %
Epoch: 122, Loss: 1.522, Test accuracy: 27.3 %
Epoch: 123, Loss: 1.421, Test accuracy: 26.4 %
Epoch: 124, Loss: 1.387, Test accuracy: 29.6 %
Epoch: 125, Loss: 1.511, Test accuracy: 28.4 %
Epoch: 126, Loss: 1.317, Test accuracy: 27.5 %
Epoch: 127, Loss: 1.441, Test accuracy: 28.5 %
Epoch: 128, Loss: 1.563, Test accuracy: 27.0 %
Epoch: 129, Loss: 1.555, Test accuracy: 27.3 %
Epoch: 130, Loss: 1.657, Test accuracy: 28.0 %
Epoch: 131, Loss: 1.586, Test accuracy: 27.8 %
Epoch: 132, Loss: 1.699, Test accuracy: 27.0 %
Epoch: 133, Loss: 1.680, Test accuracy: 27.0 %
Epoch: 134, Loss: 1.740, Test accuracy: 28.3 %
Epoch: 135, Loss: 1.700, Test accuracy: 27.8 %
Epoch: 136, Loss: 1.753, Test accuracy: 28.2 %
Epoch: 137, Loss: 1.695, Test accuracy: 27.4 %
Epoch: 138, Loss: 1.805, Test accuracy: 27.8 %
Epoch: 139, Loss: 1.750, Test accuracy: 29.4 %
Epoch: 140, Loss: 1.368, Test accuracy: 27.3 %
Epoch: 141, Loss: 1.360, Test accuracy: 28.3 %
Epoch: 142, Loss: 1.813, Test accuracy: 27.4 %
Epoch: 143, Loss: 1.886, Test accuracy: 26.3 %
Epoch: 144, Loss: 1.648, Test accuracy: 26.4 %
Epoch: 145, Loss: 1.697, Test accuracy: 27.1 %
Epoch: 146, Loss: 1.738, Test accuracy: 26.5 %
Epoch: 147, Loss: 1.632, Test accuracy: 27.4 %
Epoch: 148, Loss: 1.544, Test accuracy: 27.9 %
Epoch: 149, Loss: 1.617, Test accuracy: 27.9 %
Epoch: 150, Loss: 1.990, Test accuracy: 28.1 %
Epoch: 151, Loss: 1.429, Test accuracy: 27.1 %
Epoch: 152, Loss: 1.528, Test accuracy: 27.8 %
Epoch: 153, Loss: 1.770, Test accuracy: 27.0 %
Epoch: 154, Loss: 1.520, Test accuracy: 28.9 %
Epoch: 155, Loss: 1.676, Test accuracy: 29.0 %
Epoch: 156, Loss: 1.479, Test accuracy: 27.5 %
Epoch: 157, Loss: 1.372, Test accuracy: 26.8 %
Epoch: 158, Loss: 1.671, Test accuracy: 28.7 %
Epoch: 159, Loss: 1.731, Test accuracy: 28.9 %
Epoch: 160, Loss: 1.591, Test accuracy: 27.8 %
Epoch: 161, Loss: 1.742, Test accuracy: 28.4 %
Epoch: 162, Loss: 1.343, Test accuracy: 28.1 %
Epoch: 163, Loss: 1.528, Test accuracy: 28.0 %
Epoch: 164, Loss: 1.739, Test accuracy: 27.8 %
Epoch: 165, Loss: 1.464, Test accuracy: 26.3 %
Epoch: 166, Loss: 1.478, Test accuracy: 27.1 %
Epoch: 167, Loss: 1.238, Test accuracy: 26.5 %
Epoch: 168, Loss: 1.699, Test accuracy: 28.1 %
Epoch: 169, Loss: 1.670, Test accuracy: 28.4 %
Epoch: 170, Loss: 1.644, Test accuracy: 27.1 %
Epoch: 171, Loss: 1.444, Test accuracy: 26.2 %
Epoch: 172, Loss: 1.377, Test accuracy: 27.9 %
Epoch: 173, Loss: 1.768, Test accuracy: 28.3 %
Epoch: 174, Loss: 1.426, Test accuracy: 28.1 %
Epoch: 175, Loss: 1.827, Test accuracy: 28.4 %
Epoch: 176, Loss: 1.618, Test accuracy: 28.5 %
Epoch: 177, Loss: 1.710, Test accuracy: 28.4 %
Epoch: 178, Loss: 1.489, Test accuracy: 28.4 %
Epoch: 179, Loss: 1.704, Test accuracy: 28.3 %
Epoch: 180, Loss: 1.801, Test accuracy: 27.8 %
Epoch: 181, Loss: 1.723, Test accuracy: 28.5 %
Epoch: 182, Loss: 1.643, Test accuracy: 29.1 %
Epoch: 183, Loss: 1.391, Test accuracy: 28.1 %
Epoch: 184, Loss: 1.513, Test accuracy: 27.9 %
Epoch: 185, Loss: 1.334, Test accuracy: 27.8 %
Epoch: 186, Loss: 1.528, Test accuracy: 26.5 %
Epoch: 187, Loss: 1.662, Test accuracy: 27.3 %
Epoch: 188, Loss: 1.714, Test accuracy: 28.6 %
Epoch: 189, Loss: 1.332, Test accuracy: 27.4 %
Epoch: 190, Loss: 1.751, Test accuracy: 27.8 %
Epoch: 191, Loss: 1.787, Test accuracy: 27.9 %
Epoch: 192, Loss: 1.590, Test accuracy: 28.9 %
Epoch: 193, Loss: 1.967, Test accuracy: 28.1 %
Epoch: 194, Loss: 1.372, Test accuracy: 27.8 %
Epoch: 195, Loss: 1.561, Test accuracy: 27.4 %
Epoch: 196, Loss: 1.472, Test accuracy: 29.3 %
Epoch: 197, Loss: 1.753, Test accuracy: 28.6 %
Epoch: 198, Loss: 1.673, Test accuracy: 26.4 %
Epoch: 199, Loss: 1.521, Test accuracy: 28.1 %
Epoch: 200, Loss: 1.806, Test accuracy: 29.4 %
Epoch: 201, Loss: 1.406, Test accuracy: 26.7 %
Epoch: 202, Loss: 1.589, Test accuracy: 27.2 %
Epoch: 203, Loss: 1.374, Test accuracy: 28.3 %
Epoch: 204, Loss: 1.657, Test accuracy: 28.5 %
Epoch: 205, Loss: 1.353, Test accuracy: 27.5 %
Epoch: 206, Loss: 1.393, Test accuracy: 28.0 %
Epoch: 207, Loss: 1.307, Test accuracy: 27.0 %
Epoch: 208, Loss: 1.334, Test accuracy: 28.7 %
Epoch: 209, Loss: 1.883, Test accuracy: 27.0 %
Epoch: 210, Loss: 1.761, Test accuracy: 27.8 %
Epoch: 211, Loss: 1.838, Test accuracy: 28.3 %
Epoch: 212, Loss: 1.400, Test accuracy: 27.5 %
Epoch: 213, Loss: 1.648, Test accuracy: 27.4 %
Epoch: 214, Loss: 1.625, Test accuracy: 28.4 %
Epoch: 215, Loss: 1.504, Test accuracy: 28.3 %
Epoch: 216, Loss: 1.541, Test accuracy: 28.1 %
Epoch: 217, Loss: 1.548, Test accuracy: 27.4 %
Epoch: 218, Loss: 1.555, Test accuracy: 29.2 %
Epoch: 219, Loss: 1.718, Test accuracy: 29.1 %
Epoch: 220, Loss: 1.727, Test accuracy: 27.9 %
Epoch: 221, Loss: 1.523, Test accuracy: 28.0 %
Epoch: 222, Loss: 1.523, Test accuracy: 30.6 %
Epoch: 223, Loss: 1.429, Test accuracy: 28.2 %
Epoch: 224, Loss: 1.719, Test accuracy: 28.3 %
Epoch: 225, Loss: 1.830, Test accuracy: 27.0 %
Epoch: 226, Loss: 1.784, Test accuracy: 27.6 %
Epoch: 227, Loss: 1.614, Test accuracy: 28.7 %
Epoch: 228, Loss: 1.401, Test accuracy: 27.4 %
Epoch: 229, Loss: 1.529, Test accuracy: 29.0 %
Epoch: 230, Loss: 1.676, Test accuracy: 28.3 %
Epoch: 231, Loss: 1.829, Test accuracy: 28.7 %
Epoch: 232, Loss: 1.473, Test accuracy: 29.4 %
Epoch: 233, Loss: 1.451, Test accuracy: 27.5 %
Epoch: 234, Loss: 1.709, Test accuracy: 27.6 %
Epoch: 235, Loss: 1.631, Test accuracy: 27.4 %
Epoch: 236, Loss: 1.680, Test accuracy: 28.5 %
Epoch: 237, Loss: 1.715, Test accuracy: 27.9 %
Epoch: 238, Loss: 1.603, Test accuracy: 27.0 %
Epoch: 239, Loss: 1.642, Test accuracy: 27.4 %
Epoch: 240, Loss: 1.556, Test accuracy: 28.2 %
Epoch: 241, Loss: 1.936, Test accuracy: 28.9 %
Epoch: 242, Loss: 1.769, Test accuracy: 28.3 %
Epoch: 243, Loss: 1.576, Test accuracy: 27.1 %
Epoch: 244, Loss: 1.732, Test accuracy: 28.3 %
Epoch: 245, Loss: 1.690, Test accuracy: 28.9 %
Epoch: 246, Loss: 1.914, Test accuracy: 28.9 %
Epoch: 247, Loss: 1.454, Test accuracy: 28.2 %
Epoch: 248, Loss: 1.399, Test accuracy: 29.8 %
Epoch: 249, Loss: 1.618, Test accuracy: 25.2 %
Epoch: 250, Loss: 1.682, Test accuracy: 28.5 %
Total time for training :  1483.5314133167267





PointNetBasic
train_ds = PointCloudData("../data/ModelNet10_PLY")
test_ds = PointCloudData("../data/ModelNet10_PLY", folder='test')

Classes:  {0: 'bathtub', 1: 'bed', 2: 'chair', 3: 'desk', 4: 'dresser', 5: 'monitor', 6: 'night_stand', 7: 'sofa', 8: 'table', 9: 'toilet'}
Train dataset size:  3991
Test dataset size:  908
Number of classes:  10
Sample pointcloud shape:  torch.Size([1024, 3])
Number of parameters in the Neural Networks:  819624
Device:  cuda:0
Epoch: 1, Loss: 0.569, Test accuracy: 55.6 %
Epoch: 2, Loss: 0.386, Test accuracy: 70.7 %
Epoch: 3, Loss: 0.644, Test accuracy: 75.6 %
Epoch: 4, Loss: 0.359, Test accuracy: 72.6 %
Epoch: 5, Loss: 0.261, Test accuracy: 83.0 %
Epoch: 6, Loss: 0.560, Test accuracy: 76.8 %
Epoch: 7, Loss: 0.226, Test accuracy: 83.5 %
Epoch: 8, Loss: 0.209, Test accuracy: 77.9 %
Epoch: 9, Loss: 0.272, Test accuracy: 81.8 %
Epoch: 10, Loss: 0.125, Test accuracy: 83.7 %
Epoch: 11, Loss: 0.573, Test accuracy: 83.8 %
Epoch: 12, Loss: 0.110, Test accuracy: 83.4 %
Epoch: 13, Loss: 0.141, Test accuracy: 81.1 %
Epoch: 14, Loss: 0.228, Test accuracy: 79.6 %
Epoch: 15, Loss: 0.361, Test accuracy: 81.2 %
Epoch: 16, Loss: 0.827, Test accuracy: 84.3 %
Epoch: 17, Loss: 0.306, Test accuracy: 82.8 %
Epoch: 18, Loss: 0.563, Test accuracy: 86.7 %
Epoch: 19, Loss: 0.369, Test accuracy: 82.6 %
Epoch: 20, Loss: 0.138, Test accuracy: 82.6 %
Epoch: 21, Loss: 0.657, Test accuracy: 86.5 %
Epoch: 22, Loss: 0.046, Test accuracy: 88.2 %
Epoch: 23, Loss: 0.084, Test accuracy: 86.1 %
Epoch: 24, Loss: 0.070, Test accuracy: 85.9 %
Epoch: 25, Loss: 0.148, Test accuracy: 88.0 %
Epoch: 26, Loss: 0.071, Test accuracy: 87.3 %
Epoch: 27, Loss: 0.149, Test accuracy: 85.6 %
Epoch: 28, Loss: 0.131, Test accuracy: 86.1 %




PointNetBasic
train_ds = PointCloudData("../data/ModelNet40_PLY")
test_ds = PointCloudData("../data/ModelNet40_PLY", folder='test')

Classes:  {0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf', 5: 'bottle', 6: 'bowl', 7: 'car', 8: 'chair', 9: 'cone', 10: 'cup', 11: 'curtain', 12: 'desk', 13: 'door', 14: 'dresser', 15: 'flower_pot', 16: 'glass_box', 17: 'guitar', 18: 'keyboard', 19: 'lamp', 20: 'laptop', 21: 'mantel', 22: 'monitor', 23: 'night_stand', 24: 'person', 25: 'piano', 26: 'plant', 27: 'radio', 28: 'range_hood', 29: 'sink', 30: 'sofa', 31: 'stairs', 32: 'stool', 33: 'table', 34: 'tent', 35: 'toilet', 36: 'tv_stand', 37: 'vase', 38: 'wardrobe', 39: 'xbox'}
Train dataset size:  9843
Test dataset size:  2468
Number of classes:  40
Sample pointcloud shape:  torch.Size([1024, 3])
Number of parameters in the Neural Networks:  819624
Device:  cuda:0
Epoch: 1, Loss: 1.369, Test accuracy: 50.6 %
Epoch: 2, Loss: 0.591, Test accuracy: 57.6 %
Epoch: 3, Loss: 0.760, Test accuracy: 61.1 %
Epoch: 4, Loss: 0.737, Test accuracy: 70.6 %
Epoch: 5, Loss: 1.074, Test accuracy: 71.9 %
Epoch: 6, Loss: 1.161, Test accuracy: 76.5 %
Epoch: 7, Loss: 0.632, Test accuracy: 70.2 %
Epoch: 8, Loss: 1.644, Test accuracy: 69.2 %
Epoch: 9, Loss: 1.286, Test accuracy: 71.7 %
Epoch: 10, Loss: 0.703, Test accuracy: 75.9 %
Epoch: 11, Loss: 0.951, Test accuracy: 76.7 %
Epoch: 12, Loss: 1.210, Test accuracy: 74.2 %
Epoch: 13, Loss: 0.414, Test accuracy: 78.3 %
Epoch: 14, Loss: 0.853, Test accuracy: 78.3 %
Epoch: 15, Loss: 1.267, Test accuracy: 76.2 %
Epoch: 16, Loss: 0.283, Test accuracy: 72.5 %
Epoch: 17, Loss: 0.753, Test accuracy: 77.6 %
Epoch: 18, Loss: 0.584, Test accuracy: 78.0 %
Epoch: 19, Loss: 0.616, Test accuracy: 79.0 %
Epoch: 20, Loss: 0.911, Test accuracy: 79.2 %
Epoch: 21, Loss: 0.494, Test accuracy: 81.7 %
Epoch: 22, Loss: 0.795, Test accuracy: 79.5 %
Epoch: 23, Loss: 0.836, Test accuracy: 82.1 %
Epoch: 24, Loss: 0.708, Test accuracy: 82.8 %
Epoch: 25, Loss: 0.463, Test accuracy: 81.7 %