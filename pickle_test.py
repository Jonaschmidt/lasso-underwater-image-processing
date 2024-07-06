from graphics import Graphic

import pickle

obj = Graphic(file_path="./test_graphics/10008.png")

with open('./test_graphics/pickled.pkl', 'wb') as f:
    pickle.dump(obj, f)

print("Object saved successfully!")

with open('./test_graphics/pickled.pkl', 'rb') as f:
    loaded_obj = pickle.load(f)

print("Object loaded successfully!")
loaded_obj.show()
