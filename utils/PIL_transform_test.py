from PIL import Image, ImageOps
import numpy as np
import random

def transform_matrix_offset_center(matrix, x, y):
	o_x = float(x) / 2 + 0.5
	o_y = float(y) / 2 + 0.5
	offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
	reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
	transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
	return transform_matrix

# if __name__ == '__main__':
original_image = Image.open('/home/kyungmin/Downloads/star.jpg')
w, h = original_image.size


# rotation
rt = 10
# rt = None
if rt:
	theta = np.pi / 180 * random.uniform(-rt, rt)
else:
	theta = 0

# shear
sh=0.1
# sh = None
if sh:
	shear = random.uniform(-sh, sh)
else:
	shear = 0

# zoom
zm=[0.95, 1.05]
if zm[0] == 1 and zm[1] == 1:
	zx, zy = 1, 1
else:
	zx = random.uniform(zm[0], zm[1])
	zy = random.uniform(zm[0], zm[1])

transform_matrix = None
if theta != 0:
	rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
	transform_matrix = rotation_matrix

print (transform_matrix)
if shear != 0:
	if random.random() < 0.5:
		shear_matrix = np.array([[1, shear, 0],
								[0, 1, 0],
								[0, 0, 1]])
	else:
		shear_matrix = np.array([[1, 0, 0],
								[shear, 1, 0],
								[0, 0, 1]])
	transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

print (transform_matrix)

if zx != 1 or zy != 1:
	zoom_matrix = np.array([[zx, 0, 0],
							[0, zy, 0],
							[0, 0, 1]])
	transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

print (transform_matrix)

if transform_matrix is not None:
	transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
	transformed = original_image.transform(original_image.size,Image.AFFINE,tuple(transform_matrix.flat[:6]))
else:
	transformed = original_image

transformed.show()
