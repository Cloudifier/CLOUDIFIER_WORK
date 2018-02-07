import json
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

base_path = 'E:\\Forms\\'
dir = '00aad57c-0aed-4b94-8324-fb3eb1e696b3'

window_width = 800
window_height = 600
window_size_default_value = 120
lbls = ['Button', 'ComboBox', 'RadioButton', 'CheckBox', 'TextBox', 'Label', 'DatePicker']

json_path = base_path + '{}\\form_{}.json'.format(dir, dir)
el_path = base_path + '{}\\form_{}.png'.format(dir, dir)
json = json.load(open(json_path))
image = scipy.misc.imread(el_path, flatten=False, mode='RGB')
b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2] # For RGB image

class Control(object):
    def __init__(self, el):
        self.name = el['Control']
        self.xpos = int(el['XPosition'])
        self.ypos = int(el['YPosition'])
        self.width = int(el['Width']) #Should be integer from image generation
        self.height = int(el['Height']) #Should be integer from image generation
        self.red = el['Red']
        self.blue = el['Blue']
        self.green = el['Green']
        self.uid = el['Uid']

new_image = np.full((window_height, window_width, 3), window_size_default_value)
distinct_els = list()
for el in json:
    ctrl = Control(el)
    crt_lbl = lbls.index(ctrl.name)
    print(f'Uid:{ctrl.uid}, width:{ctrl.width}, height:{ctrl.height}, xpos:{ctrl.xpos}, ypos:{ctrl.ypos}')
    for i in range(ctrl.ypos, ctrl.ypos + ctrl.height):
        if i >= window_height:
            continue
        for j in range(ctrl.xpos, ctrl.xpos  + ctrl.width):
            if j >= window_width:
                continue
            new_image[i, j] = crt_lbl

plt.imshow(new_image[:, :, 0])
plt.show()
