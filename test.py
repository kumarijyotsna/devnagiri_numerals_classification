
from keras.models import model_from_json
import numpy as np
# load json and create model
json_file = open('model_s.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_s.h5")
print("Loaded model from disk")
from keras.preprocessing import image

test_image = image.load_img('c.png', target_size = (64, 64))
print test_image
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
print (result)


