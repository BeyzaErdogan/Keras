#El yazısı rakamlarını sınıflandırma

import tensorflow as t
from keras.datasets import mnist #internet verisetini yükleyelim

(x_train,y_train), (x_test,y_test) = mnist.load_data()

print(x_train.shape) #(60000,28,28) ->60000 veri 28x28lik

from keras import layers #layerlar sinir ağlarının tabakalarıdır. Her bir tabakada öğrenme gerçekleşir.

from keras import models 

network = models.Sequential()#katmanlı model oluşturulur.
network.add(layers.Dense(512,activation="relu",input_shape=(28*28,))) #katmanllardaki neronları(bir önceki katmandaki neronları bağlamak için) birbirine bağlamak için

network.add(layers.Dense(10,activation="softmax"))

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
#veriler 0 ile 255 arasında değer alıyor bunları 0-1 arasında ve floata çevrilir

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

#etiketleri kategorik olarak kodlama
#onehatencoder aslında etiketleri sütunlara çevirmekte.
from keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

#sinir ağını design ettikten sonra compile(derleme) komutu ile model yapılandırılır.
network.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])

network.fit(x_train,y_train,epochs=5, batch_size=128)


network.evaluate(x_test,y_test)