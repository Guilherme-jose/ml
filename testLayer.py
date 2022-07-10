import layer
import numpy as np
from PIL import Image

image = Image.open('training_set_small/cats/cat.1.jpg')
data = np.matrix(list(image.getdata()))
image.close()
data = np.reshape(data, (75,70))
con = layer.kernelLayer(3, 75)
max = layer.maxPoolLayer(75)
result = con.forward(data)
result  = max.forward(result)

result = Image.fromarray(result)
result.show()