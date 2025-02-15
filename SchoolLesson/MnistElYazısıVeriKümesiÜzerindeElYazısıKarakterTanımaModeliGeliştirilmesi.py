import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.legacy import Adam  # Python 3.12 uyumluluğu için legacy Adam kullanımı

# MNIST veri kümesini yükleme
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Veri kümesini şekil ve içerik olarak inceleme
print("Eğitim veri kümesi boyutu:", train_images.shape)
print("Test veri kümesi boyutu:", test_images.shape)
print("Örnek etiketler:", np.unique(train_labels))

# 2000 adet eğitim ve test verisi seçme ve normalizasyon
train_images, train_labels = train_images[:2000] / 255.0, train_labels[:2000]
test_images, test_labels = test_images[:2000] / 255.0, test_labels[:2000]

# Eğitim verilerinin ilk birkaçını görselleştirme
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(str(train_labels[i]))
plt.show()

# Modeli tanımlama
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Giriş katmanı (28x28 matrisi düzleştirme)
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),  # Daha büyük gizli katman
    tf.keras.layers.BatchNormalization(),  # Batch Normalization ekleme
    tf.keras.layers.Dropout(0.3),  # Aşırı öğrenmeyi önlemek için dropout
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')  # Çıktı katmanı (10 sınıf)
])

# Modelin derlenmesi
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
history = model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels), batch_size=32)

# Modeli test etme
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Eğitim süreci görselleştirme
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.title('Eğitim ve Doğrulama Doğruluğu')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.title('Eğitim ve Doğrulama Kaybı')
plt.show()

# Modelin tahmin yapması
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# İlk birkaç tahmini görselleştirme
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f'Gerçek: {test_labels[i]}\nTahmin: {predicted_labels[i]}')
plt.show()
