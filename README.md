使用Gradient Descent實作mnist(20180623)
==========================================
首先為載入這次實作回用到的模塊
<pre><code>import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils</pre></code>
原本我是想讓keras直接幫我從網路上下載mnist的database，使用如下
<pre><code>from keras.datasets import mnist</pre></code>
結果下載速度極慢，而且容易下載到一半卡住<br/>
於是自己從網路上找下載檔案<br/>
<http://yann.lecun.com/exdb/mnist/> <br/>
好處是速度快多了，壞處是不能直接使用一般大家使用的程式來載入資料
<pre><code>(x_train, y_train), (X_test, y_test) = mnist.load_data()</pre></code>
檔案格式為gz Archive (.gz)，於是又自己上網找了一下用python開檔的方式，首先載入gzip模塊(跟開圖片或文字檔方式很像)
<pre><code>import gzip</pre></code>
接著依序將4個檔案指定給x_train、y_train、x_test、y_test
<pre><code>with gzip.open('C:\python36\Lib\site-packages\keras\datasets\mnist.npz/train-images-idx3-ubyte.gz', 'rb') as f1:
    x_train=f1.read()
with gzip.open('C:\python36\Lib\site-packages\keras\datasets\mnist.npz/train-labels-idx1-ubyte.gz',  'rb') as f2:
    y_train=f2.read()
with gzip.open('C:\python36\Lib\site-packages\keras\datasets\mnist.npz/t10k-images-idx3-ubyte.gz', 'rb') as f3:
    x_test=f3.read()
with gzip.open('C:\python36\Lib\site-packages\keras\datasets\mnist.npz/t10k-labels-idx1-ubyte.gz',  'rb') as f4:
    y_test=f4.read()</pre></code>
為了要將每張圖片變成28x28的pixel傳入我的neural network，必須先reshape這筆資料 <br/>
但卻發生錯誤，原因是這筆x_train的type為bytes，同時資料的開頭處並不是我所想要的 <br/>
資細看了下MNIST handwritten官網上label的資料:
<pre><code>TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.</pre></code>
還有官網上image的資料:
<pre><code>TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).</pre></code>
於是先將x_train去掉前面部分，並且轉array然後reshape成60000張28*28的圖片
<pre><code>from numpy import *
x_train=bytearray(x_train)
x_train=x_train[16:47040016]
x_train=array(x_train)
x_train=x_train.reshape(60000,28*28)</pre></code>
同理將x_test去掉前面部分，並且轉array然後reshape成10000張28*28的圖片
<pre><code>x_test=bytearray(x_test)
x_test=x_test[16:7840016]
x_test=array(x_test)
x_test=x_test.reshape(10000,28*28)</pre></code>
接著將y_train去掉前面部分，並且轉array然後每個元素轉成10維的向量
<pre><code>y_train=y_train[8:60008]
y_train=bytearray(y_train)
y_train=array(y_train)
y_train = np_utils.to_categorical( y_train, num_classes=10)</pre></code>
同理將y_test去掉前面部分，並且轉array然後每個元素轉成10維的向量
<pre><code>y_test=y_test[8:10008]
y_test=bytearray(y_test)
y_test=array(y_test)
y_test = np_utils.to_categorical( y_test, num_classes=10)</pre></code>
因為image的每個pixel有255的深淺數值，先進行normalize，變成0到1的值
<pre><code>x_train=x_train/255
x_test=x_test/255</pre></code>
接著才開始程式，沒想到資料處理花了我這麼多時間....
首先宣告一個model
<pre><code>model=Sequential()</pre></code>
開始加入第一層，使用Full-connected Layer，輸入為image維度是28x28，輸出為50維的vector，並使用sigmoid激活函數
<pre><code>model.add(Dense(input_dim=28*28,units=50,activation='sigmoid'))</pre></code>
然後再加入個幾層....輸出都是50維的vector，並使用sigmoid激活函數
<pre><code>model.add(Dense(units=50,activation='sigmoid'))
model.add(Dense(units=50,activation='sigmoid'))
model.add(Dense(units=50,activation='sigmoid'))</pre></code>
最後一層輸出10維的vector，就是對應了1到10的數字，使用softmax激活函數
<pre><code>model.add(Dense(units=10,activation='softmax'))</pre></code>
編譯動作，Loss function使用mean-square-error，使用Gradient Descent優化，learning rate=0.01，並監看他的準確率
<pre><code>model.compile(loss='mse',optimizer=SGD(lr=0.01),metrics=['accuracy'])</pre></code>
輸入訓練的data，每個batch內有32個training data，重複執行所有batch共2次
<pre><code>model.fit(x_train,y_train,batch_size=32,epochs=2)</pre></code>
![Imgur](https://i.imgur.com/nKnuaCU.png)
最後結果:
<pre><code>60000/60000 [==============================] - 301s 5ms/step - loss: 0.0908 - acc: 0.1124</pre></code>
準確率只有11.24%簡直慘不忍睹....10個猜1個數，運氣好一點都比它準了...<br/>
看來還有不少問題待優化~~

後記(20180626)
=======
考量到可能因為使用了sigmoid激活函數，造成越靠近input端的gradient較小，幾乎無法對輸出造成影響<br/>
所以把激活函數改成了ReLU做測試
<pre><code>model.add(Dense(units=50,activation='relu'))</pre></code>
並且計算Loss function的方式改用cross entropy，
<pre><code>model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.001),metrics=['accuracy'])</pre></code>
則測試結果如下:
<pre><code>60000/60000 [==============================] - 311s 5ms/step - loss: 1.5986 - acc: 0.5923</pre></code>
對於training data的正確率大幅上升了，但遺憾的是loss也增加了




          
