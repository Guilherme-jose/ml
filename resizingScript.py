from PIL import Image
import numpy as np

for i in range(4000):
    load_img_rz = np.array(Image.open('training_set/cats/cat.'+ str(i+1) +'.jpg').resize((75,70)).convert('L'))
    Image.fromarray(load_img_rz).save('training_set_small/cats/cat.'+ str(i+1) +'.jpg')
    print("After resizing:",load_img_rz.shape)