import layer
import numpy as np
from PIL import Image

image = Image.open('training_set_small/cats/cat.1.jpg')
data = np.matrix(list(image.getdata()))
image.close()
data = np.reshape(data, (-1,1))
con = layer.flattenLayer(75, 5250)
max = layer.widenLayer(75)
#result = con.forward(data)
result  = max.forward(data)
#result = con.forward(data)
#result  = max.forward(result)

result = Image.fromarray(result)
result.show()