import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from my_models.train_model import traning_model

model_name = "example"
degits_to_test = ['1','2','3','9']
def exist_file(file):
    return os.path.isfile(f"\models\\{file}.keras")
    

def main():
    if exist_file(model_name):
        print(f"Нету готовой модели. \nСейчас мы обучим!")
        traning_model()
    model = load_model(f"\models\\{model_name}.keras")
    for deg in degits_to_test:
        img = image.load_img(f'datasets/numbers/num_{deg}.png', target_size=(28,28), color_mode='grayscale')
        x = image.img_to_array(img) / 255.0
        x = x.reshape(1,28,28,1)
        pred = model.predict(x)
        print("Предсказанная цифра:", np.argmax(pred),' с вероятностью: ', pred[0][np.argmax(pred)])
        
    