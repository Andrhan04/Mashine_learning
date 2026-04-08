import os
import sys
import warnings
import logging

# Сохраняем оригинальный stderr
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Максимальное отключение всего
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_LOGGING_LEVEL'] = 'FATAL'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

warnings.filterwarnings('ignore')

# Настраиваем логирование
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)
logging.getLogger('h5py').setLevel(logging.FATAL)
logging.getLogger('PIL').setLevel(logging.FATAL)

# Импортируем TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(0)

# Восстанавливаем stderr
sys.stderr = original_stderr

import os
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Отключаем GPU, чтобы TensorFlow работал только на CPU

model_name = "example"
def exist_file(file):
    return os.path.isfile(f"my_models\\{file}.keras")


def main():
    if not exist_file(model_name):
        from my_models.train_model import traning_model
        print(f"Нету готовой модели. \nСейчас мы обучим!")
        traning_model()
        print("Модель обучена")
    else:
        print("Модель уже есть, загружаю")
    model = load_model(f"my_models\\{model_name}.keras")
    
    img = image.load_img(f'images/num_9.png', target_size=(28,28), color_mode='grayscale')
    x = image.img_to_array(img) / 255.0
    x = x.reshape(1,28,28,1)
    pred = model.predict(x)
    print("Предсказанная цифра:", np.argmax(pred),' с вероятностью: ', pred[0][np.argmax(pred)])
        
main()