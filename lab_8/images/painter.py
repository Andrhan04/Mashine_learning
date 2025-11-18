import matplotlib.pyplot as plt


def draw_dataset(X_train, y_train):
    # Как выглядит датасет
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([]) # Убираем деления по ОХ
        plt.yticks([]) # Убираем деления по ОУ
        plt.grid(False)
        plt.imshow(X_train[i].reshape(28, 28), 'gray')
        plt.xlabel(y_train[i])
    plt.savefig('images\\dataset')
    plt.show()
    
def draw_accuracy(history):
    # Точность
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Точность обучения')
    plt.legend()
    plt.savefig('images\\accuratly')
    plt.show()

def draw_loss(history):
    # Потери
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Потери обучения')
    plt.legend()
    plt.savefig('images\\loss')
    plt.show()