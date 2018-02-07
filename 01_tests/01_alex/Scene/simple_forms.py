import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
np.set_printoptions(threshold=np.nan)

base_path = 'E:\\SimpleForms\\'
dir = '6d2dd43e-bbf0-40ca-93a7-bb40232e7ba9'
el = '84db23c4-63c8-40f1-a270-2ab49183ec1f.png'

window_width = 250
window_height = 100
window_size_default_value = 120

json_path = base_path + '{}\\form_{}.json'.format(dir, dir)
el_path = base_path + '{}\\{}'.format(dir, el)
data = json.load(open(json_path))
image = scipy.misc.imread(el_path, flatten=False, mode='RGB')
b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2] # For RGB image

img_width = int(data[0]['Width'])
img_height = int(data[0]['Height'])
print(img_height, img_width)

red_channel = np.reshape(data[0]['Red'], (int(img_width), int(img_height))).T
green_channel = np.reshape(data[0]['Green'], (int(img_width), int(img_height))).T
blue_channel = np.reshape(data[0]['Blue'], (int(img_width), int(img_height))).T


arr = np.full((window_height, window_width, 3), window_size_default_value)
width_center = int((window_width - img_width) / 2)
height_center = int((window_height - img_height) / 2)
arr[height_center: height_center + img_height, width_center:width_center + img_width] = 1.0
print(arr)
plt.imshow(arr[:,:,0])
plt.show()



