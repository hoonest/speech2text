# 0. 사용할 패키지 불러오기
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
import os
import speech_data

# 랜덤시드 고정시키기
np.random.seed(5)


# 1. 데이터 준비하기
#=====================================================
batch=speech_data.wave_batch_generator(1205,target=speech_data.Target.digits)
X,Y=next(batch)
x_train = np.array(X)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))
y_train = np.array(Y)
one_hot_vec_size = y_train.shape[1]

# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))




print("one hot encoding vector size is ", one_hot_vec_size)

print("x_train, ytrain vector size is ", x_train.shape, "=====", y_train.shape)

# 3. 모델 구성하기
model = Sequential()
model.add(LSTM(128, input_shape=(8192,1)))
model.add(Dense(one_hot_vec_size, activation='softmax'))

# 4. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = LossHistory()  # 손실 이력 객체 생성
history.init()

# 5. 모델 학습시키기
model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=2, callbacks=[history])

# 6. 학습과정 살펴보기
# % matplotlib inline
import matplotlib.pyplot as plt

plt.plot(history.losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# 7. 모델 평가하기
scores = model.evaluate(x_train, y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# 8. 모델 사용하기

path = "data/number_test/"
files = os.listdir(path)

for wav in files:
    demo_file = path + wav
    demo = speech_data.load_wav_file(demo_file)
    result = model.predict(np.array([demo]))
    result = np.argmax(result)
    print("== predicted digit for %s :result = %d " % (demo_file, result))