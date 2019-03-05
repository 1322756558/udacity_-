
## 卷积神经网络（Convolutional Neural Network, CNN）

## 项目：实现一个狗品种识别算法App

在这个notebook文件中，有些模板代码已经提供给你，但你还需要实现更多的功能来完成这个项目。除非有明确要求，你无须修改任何已给出的代码。以**'(练习)'**开始的标题表示接下来的代码部分中有你需要实现的功能。这些部分都配有详细的指导，需要实现的部分也会在注释中以'TODO'标出。请仔细阅读所有的提示。

除了实现代码外，你还**需要**回答一些与项目及代码相关的问题。每个需要回答的问题都会以 **'问题 X'** 标记。请仔细阅读每个问题，并且在问题后的 **'回答'** 部分写出完整的答案。我们将根据 你对问题的回答 和 撰写代码实现的功能 来对你提交的项目进行评分。

>**提示：**Code 和 Markdown 区域可通过 **Shift + Enter** 快捷键运行。此外，Markdown可以通过双击进入编辑模式。

项目中显示为_选做_的部分可以帮助你的项目脱颖而出，而不是仅仅达到通过的最低要求。如果你决定追求更高的挑战，请在此 notebook 中完成_选做_部分的代码。

---

### 让我们开始吧
在这个notebook中，你将迈出第一步，来开发可以作为移动端或 Web应用程序一部分的算法。在这个项目的最后，你的程序将能够把用户提供的任何一个图像作为输入。如果可以从图像中检测到一只狗，它会输出对狗品种的预测。如果图像中是一个人脸，它会预测一个与其最相似的狗的种类。下面这张图展示了完成项目后可能的输出结果。（……实际上我们希望每个学生的输出结果不相同！）

![Sample Dog Output](images/sample_dog_output.png)

在现实世界中，你需要拼凑一系列的模型来完成不同的任务；举个例子，用来预测狗种类的算法会与预测人类的算法不同。在做项目的过程中，你可能会遇到不少失败的预测，因为并不存在完美的算法和模型。你最终提交的不完美的解决方案也一定会给你带来一个有趣的学习经验！

### 项目内容

我们将这个notebook分为不同的步骤，你可以使用下面的链接来浏览此notebook。

* [Step 0](#step0): 导入数据集
* [Step 1](#step1): 检测人脸
* [Step 2](#step2): 检测狗狗
* [Step 3](#step3): 从头创建一个CNN来分类狗品种
* [Step 4](#step4): 使用一个CNN来区分狗的品种(使用迁移学习)
* [Step 5](#step5): 建立一个CNN来分类狗的品种（使用迁移学习）
* [Step 6](#step6): 完成你的算法
* [Step 7](#step7): 测试你的算法

在该项目中包含了如下的问题：

* [问题 1](#question1)
* [问题 2](#question2)
* [问题 3](#question3)
* [问题 4](#question4)
* [问题 5](#question5)
* [问题 6](#question6)
* [问题 7](#question7)
* [问题 8](#question8)
* [问题 9](#question9)
* [问题 10](#question10)
* [问题 11](#question11)


---
<a id='step0'></a>
## 步骤 0: 导入数据集

### 导入狗数据集
在下方的代码单元（cell）中，我们导入了一个狗图像的数据集。我们使用 scikit-learn 库中的 `load_files` 函数来获取一些变量：
- `train_files`, `valid_files`, `test_files` - 包含图像的文件路径的numpy数组
- `train_targets`, `valid_targets`, `test_targets` - 包含独热编码分类标签的numpy数组
- `dog_names` - 由字符串构成的与标签相对应的狗的种类


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('/data/dog_images/train')
valid_files, valid_targets = load_dataset('/data/dog_images/valid')
test_files, test_targets = load_dataset('/data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    Using TensorFlow backend.
    

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.
    

### 导入人脸数据集

在下方的代码单元中，我们导入人脸图像数据集，文件所在路径存储在名为 `human_files` 的 numpy 数组。


```python
import random
random.seed(8675309)

# 加载打乱后的人脸数据集的文件名
human_files = np.array(glob("/data/lfw/*/*"))
random.shuffle(human_files)

# 打印数据集的数据量
print('There are %d total human images.' % len(human_files))
```

    There are 13233 total human images.
    

---
<a id='step1'></a>
## 步骤1：检测人脸
 
我们将使用 OpenCV 中的 [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) 来检测图像中的人脸。OpenCV 提供了很多预训练的人脸检测模型，它们以XML文件保存在 [github](https://github.com/opencv/opencv/tree/master/data/haarcascades)。我们已经下载了其中一个检测模型，并且把它存储在 `haarcascades` 的目录中。

在如下代码单元中，我们将演示如何使用这个检测模型在样本图像中找到人脸。


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# 提取预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# 加载彩色（通道顺序为BGR）图像
img = cv2.imread(human_files[3])

# 将BGR图像进行灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 在图像中找出脸
faces = face_cascade.detectMultiScale(gray)

# 打印图像中检测到的脸的个数
print('Number of faces detected:', len(faces))

# 获取每一个所检测到的脸的识别框
for (x,y,w,h) in faces:
    # 在人脸图像中绘制出识别框
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# 将BGR图像转变为RGB图像以打印
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 展示含有识别框的图像
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1
    


![png](output_5_1.png)


在使用任何一个检测模型之前，将图像转换为灰度图是常用过程。`detectMultiScale` 函数使用储存在 `face_cascade` 中的的数据，对输入的灰度图像进行分类。

在上方的代码中，`faces` 以 numpy 数组的形式，保存了识别到的面部信息。它其中每一行表示一个被检测到的脸，该数据包括如下四个信息：前两个元素  `x`、`y` 代表识别框左上角的 x 和 y 坐标（参照上图，注意 y 坐标的方向和我们默认的方向不同）；后两个元素代表识别框在 x 和 y 轴两个方向延伸的长度 `w` 和 `d`。 

### 写一个人脸识别器

我们可以将这个程序封装为一个函数。该函数的输入为人脸图像的**路径**，当图像中包含人脸时，该函数返回 `True`，反之返回 `False`。该函数定义如下所示。


```python
# 如果img_path路径表示的图像检测到了脸，返回"True" 
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### **【练习】** 评估人脸检测模型


---

<a id='question1'></a>
### __问题 1:__ 

在下方的代码块中，使用 `face_detector` 函数，计算：

- `human_files` 的前100张图像中，能够检测到**人脸**的图像占比多少？
- `dog_files` 的前100张图像中，能够检测到**人脸**的图像占比多少？

理想情况下，人图像中检测到人脸的概率应当为100%，而狗图像中检测到人脸的概率应该为0%。你会发现我们的算法并非完美，但结果仍然是可以接受的。我们从每个数据集中提取前100个图像的文件路径，并将它们存储在`human_files_short`和`dog_files_short`中。


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
## 请不要修改上方代码
num_human=np.mean([face_detector(human) for human in human_files_short])

num_dog=np.mean([face_detector(dog) for dog in dog_files_short])
        
print("human num is {0}%dog num is {1}%".format(num_human*100, num_dog*100))

## TODO: 基于human_files_short和dog_files_short
## 中的图像测试face_detector的表现

```

    human num is 100.0%dog num is 11.0%
    

---

<a id='question2'></a>

### __问题 2:__ 

就算法而言，该算法成功与否的关键在于，用户能否提供含有清晰面部特征的人脸图像。
那么你认为，这样的要求在实际使用中对用户合理吗？如果你觉得不合理，你能否想到一个方法，即使图像中并没有清晰的面部特征，也能够检测到人脸？

__回答:__ 合理,算法的功能为检测面部图像,那么就应当向其提供清晰的面部特征



---

<a id='Selection1'></a>
### 选做：

我们建议在你的算法中使用opencv的人脸检测模型去检测人类图像，不过你可以自由地探索其他的方法，尤其是尝试使用深度学习来解决它:)。请用下方的代码单元来设计和测试你的面部监测算法。如果你决定完成这个_选做_任务，你需要报告算法在每一个数据集上的表现。


```python
## (选做) TODO: 报告另一个面部检测算法在LFW数据集上的表现
### 你可以随意使用所需的代码单元数
```

---
<a id='step2'></a>

## 步骤 2: 检测狗狗

在这个部分中，我们使用预训练的 [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) 模型去检测图像中的狗。下方的第一行代码就是下载了 ResNet-50 模型的网络结构参数，以及基于 [ImageNet](http://www.image-net.org/) 数据集的预训练权重。

ImageNet 这目前一个非常流行的数据集，常被用来测试图像分类等计算机视觉任务相关的算法。它包含超过一千万个 URL，每一个都链接到 [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 中所对应的一个物体的图像。任给输入一个图像，该 ResNet-50 模型会返回一个对图像中物体的预测结果。


```python
from keras.applications.resnet50 import ResNet50

# 定义ResNet50模型
ResNet50_model = ResNet50(weights='imagenet')
```

    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
    102858752/102853048 [==============================] - 2s 0us/step
    

### 数据预处理

- 在使用 TensorFlow 作为后端的时候，在 Keras 中，CNN 的输入是一个4维数组（也被称作4维张量），它的各维度尺寸为 `(nb_samples, rows, columns, channels)`。其中 `nb_samples` 表示图像（或者样本）的总数，`rows`, `columns`, 和 `channels` 分别表示图像的行数、列数和通道数。


- 下方的 `path_to_tensor` 函数实现如下将彩色图像的字符串型的文件路径作为输入，返回一个4维张量，作为 Keras CNN 输入。因为我们的输入图像是彩色图像，因此它们具有三个通道（ `channels` 为 `3`）。
    1. 该函数首先读取一张图像，然后将其缩放为 224×224 的图像。
    2. 随后，该图像被调整为具有4个维度的张量。
    3. 对于任一输入图像，最后返回的张量的维度是：`(1, 224, 224, 3)`。


- `paths_to_tensor` 函数将图像路径的字符串组成的 numpy 数组作为输入，并返回一个4维张量，各维度尺寸为 `(nb_samples, 224, 224, 3)`。 在这里，`nb_samples`是提供的图像路径的数据中的样本数量或图像数量。你也可以将 `nb_samples` 理解为数据集中3维张量的个数（每个3维张量表示一个不同的图像。


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### 基于 ResNet-50 架构进行预测

对于通过上述步骤得到的四维张量，在把它们输入到 ResNet-50 网络、或 Keras 中其他类似的预训练模型之前，还需要进行一些额外的处理：
1. 首先，这些图像的通道顺序为 RGB，我们需要重排他们的通道顺序为 BGR。
2. 其次，预训练模型的输入都进行了额外的归一化过程。因此我们在这里也要对这些张量进行归一化，即对所有图像所有像素都减去像素均值 `[103.939, 116.779, 123.68]`（以 RGB 模式表示，根据所有的 ImageNet 图像算出）。

导入的 `preprocess_input` 函数实现了这些功能。如果你对此很感兴趣，可以在 [这里](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py) 查看 `preprocess_input`的代码。


在实现了图像处理的部分之后，我们就可以使用模型来进行预测。这一步通过 `predict` 方法来实现，它返回一个向量，向量的第 i 个元素表示该图像属于第 i 个 ImageNet 类别的概率。这通过如下的 `ResNet50_predict_labels` 函数实现。

通过对预测出的向量取用 argmax 函数（找到有最大概率值的下标序号），我们可以得到一个整数，即模型预测到的物体的类别。进而根据这个 [清单](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)，我们能够知道这具体是哪个品种的狗狗。



```python
from keras.applications.resnet50 import preprocess_input, decode_predictions
def ResNet50_predict_labels(img_path):
    # 返回img_path路径的图像的预测向量
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### 完成狗检测模型


在研究该 [清单](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 的时候，你会注意到，狗类别对应的序号为151-268。因此，在检查预训练模型判断图像是否包含狗的时候，我们只需要检查如上的 `ResNet50_predict_labels` 函数是否返回一个介于151和268之间（包含区间端点）的值。

我们通过这些想法来完成下方的 `dog_detector` 函数，如果从图像中检测到狗就返回 `True`，否则返回 `False`。


```python
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### 【作业】评估狗狗检测模型

---

<a id='question3'></a>
### __问题 3:__ 

在下方的代码块中，使用 `dog_detector` 函数，计算：

- `human_files_short`中图像检测到狗狗的百分比？
- `dog_files_short`中图像检测到狗狗的百分比？


```python
### TODO: 测试dog_detector函数在human_files_short和dog_files_short的表现
dd_human=np.mean([dog_detector(human) for human in human_files_short])
dd_dog=np.mean([dog_detector(dog) for dog in dog_files_short])
        
print('dog_files_short-human {0}%, dog_files_short-dog {1}%'.format(dd_human*100, dd_dog*100))
        
```

    dog_files_short-human 0.0%, dog_files_short-dog 100.0%
    

---

<a id='step3'></a>

## 步骤 3: 从头开始创建一个CNN来分类狗品种


现在我们已经实现了一个函数，能够在图像中识别人类及狗狗。但我们需要更进一步的方法，来对狗的类别进行识别。在这一步中，你需要实现一个卷积神经网络来对狗的品种进行分类。你需要__从头实现__你的卷积神经网络（在这一阶段，你还不能使用迁移学习），并且你需要达到超过1%的测试集准确率。在本项目的步骤五种，你还有机会使用迁移学习来实现一个准确率大大提高的模型。

在添加卷积层的时候，注意不要加上太多的（可训练的）层。更多的参数意味着更长的训练时间，也就是说你更可能需要一个 GPU 来加速训练过程。万幸的是，Keras 提供了能够轻松预测每次迭代（epoch）花费时间所需的函数。你可以据此推断你算法所需的训练时间。

值得注意的是，对狗的图像进行分类是一项极具挑战性的任务。因为即便是一个正常人，也很难区分布列塔尼犬和威尔士史宾格犬。


布列塔尼犬（Brittany） | 威尔士史宾格犬（Welsh Springer Spaniel）
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

不难发现其他的狗品种会有很小的类间差别（比如金毛寻回犬和美国水猎犬）。


金毛寻回犬（Curly-Coated Retriever） | 美国水猎犬（American Water Spaniel）
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">

同样，拉布拉多犬（labradors）有黄色、棕色和黑色这三种。那么你设计的基于视觉的算法将不得不克服这种较高的类间差别，以达到能够将这些不同颜色的同类狗分到同一个品种中。

黄色拉布拉多犬（Yellow Labrador） | 棕色拉布拉多犬（Chocolate Labrador） | 黑色拉布拉多犬（Black Labrador）
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

我们也提到了随机分类将得到一个非常低的结果：不考虑品种略有失衡的影响，随机猜测到正确品种的概率是1/133，相对应的准确率是低于1%的。

请记住，在深度学习领域，实践远远高于理论。大量尝试不同的框架吧，相信你的直觉！当然，玩得开心！


### 数据预处理


通过对每张图像的像素值除以255，我们对图像实现了归一化处理。


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████| 6680/6680 [01:15<00:00, 88.39it/s] 
    100%|██████████| 835/835 [00:08<00:00, 97.98it/s] 
    100%|██████████| 836/836 [00:08<00:00, 98.94it/s] 
    

### 【练习】模型架构


创建一个卷积神经网络来对狗品种进行分类。在你代码块的最后，执行 `model.summary()` 来输出你模型的总结信息。
    
我们已经帮你导入了一些所需的 Python 库，如有需要你可以自行导入。如果你在过程中遇到了困难，如下是给你的一点小提示——该模型能够在5个 epoch 内取得超过1%的测试准确率，并且能在CPU上很快地训练。

![Sample CNN](images/sample_cnn.png)

---

<a id='question4'></a>  

### __问题 4:__ 

在下方的代码块中尝试使用 Keras 搭建卷积网络的架构，并回答相关的问题。

1. 你可以尝试自己搭建一个卷积网络的模型，那么你需要回答你搭建卷积网络的具体步骤（用了哪些层）以及为什么这样搭建。
2. 你也可以根据上图提示的步骤搭建卷积网络，那么请说明为何如上的架构能够在该问题上取得很好的表现。

__回答:__ 在我自己搭建的卷积网络中使用了卷积层,最大池化层,全局平均池化层,随机断点层和全连接层,在使用卷积层是为了降低维数,使用最大池化层和全局平均池化层是为了有效的降低每一层的宽高,提高运算效率,之中采用随机断点的方式来断开某些结点,使得其他结点能够得到更有效的训练,防止模型过拟合,在卷积层中使用relu为误差函数防止梯度消失,在最后由于狗的品种为133种所以使用拥有133个结点的全连接层使用softmax误差函数
同时,在卷积层中,除了降低维数,卷积层中更重要的作用是过滤图片中的信息或特征,例如边缘,色块等
在卷积层中会有多个过滤器,每种过滤器对应一个特征,当有多个特征时他本身的堆栈会变得很大,这样容易导致过拟合
故池化层将卷积层作为输入,除了降低每一层的宽高,提高运算效率,也是为了降低卷积层特征的维数,防止过拟合的现象发生


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: 定义你的网络架构
model.add(Conv2D(filters=32, kernel_size=4, padding='same', activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=256, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(GlobalAveragePooling2D())
model.add(Dense(133, activation='softmax'))


                 
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 224, 224, 32)      1568      
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 112, 112, 32)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 112, 112, 64)      32832     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 56, 56, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 56, 56, 128)       131200    
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 28, 28, 128)       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 28, 28, 256)       524544    
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 14, 14, 256)       0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 14, 14, 256)       0         
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 133)               34181     
    =================================================================
    Total params: 724,325
    Trainable params: 724,325
    Non-trainable params: 0
    _________________________________________________________________
    


```python
## 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 【练习】训练模型


---

<a id='question5'></a>  

### __问题 5:__ 

在下方代码单元训练模型。使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。

可选题：你也可以对训练集进行 [数据增强](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)，来优化模型的表现。




```python
from keras.callbacks import ModelCheckpoint  

### TODO: 设置训练模型的epochs的数量

epochs = 10

### 不要修改下方代码

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/10
    6660/6680 [============================>.] - ETA: 0s - loss: 4.8858 - acc: 0.0125Epoch 00001: val_loss improved from inf to 4.85387, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 51s 8ms/step - loss: 4.8859 - acc: 0.0124 - val_loss: 4.8539 - val_acc: 0.0156
    Epoch 2/10
    6660/6680 [============================>.] - ETA: 0s - loss: 4.8242 - acc: 0.0165Epoch 00002: val_loss improved from 4.85387 to 4.77666, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 50s 7ms/step - loss: 4.8240 - acc: 0.0166 - val_loss: 4.7767 - val_acc: 0.0180
    Epoch 3/10
    6660/6680 [============================>.] - ETA: 0s - loss: 4.7372 - acc: 0.0201Epoch 00003: val_loss improved from 4.77666 to 4.65168, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 50s 8ms/step - loss: 4.7367 - acc: 0.0202 - val_loss: 4.6517 - val_acc: 0.0251
    Epoch 4/10
    6660/6680 [============================>.] - ETA: 0s - loss: 4.6289 - acc: 0.0308Epoch 00004: val_loss improved from 4.65168 to 4.56124, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 50s 8ms/step - loss: 4.6297 - acc: 0.0307 - val_loss: 4.5612 - val_acc: 0.0383
    Epoch 5/10
    6660/6680 [============================>.] - ETA: 0s - loss: 4.4651 - acc: 0.0468Epoch 00005: val_loss improved from 4.56124 to 4.39722, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 51s 8ms/step - loss: 4.4655 - acc: 0.0467 - val_loss: 4.3972 - val_acc: 0.0563
    Epoch 6/10
    6660/6680 [============================>.] - ETA: 0s - loss: 4.2818 - acc: 0.0640Epoch 00006: val_loss improved from 4.39722 to 4.26345, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 51s 8ms/step - loss: 4.2819 - acc: 0.0639 - val_loss: 4.2634 - val_acc: 0.0683
    Epoch 7/10
    6660/6680 [============================>.] - ETA: 0s - loss: 4.0946 - acc: 0.0892Epoch 00007: val_loss improved from 4.26345 to 4.07911, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 51s 8ms/step - loss: 4.0938 - acc: 0.0894 - val_loss: 4.0791 - val_acc: 0.0862
    Epoch 8/10
    6660/6680 [============================>.] - ETA: 0s - loss: 3.9270 - acc: 0.1027Epoch 00008: val_loss improved from 4.07911 to 3.97221, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 51s 8ms/step - loss: 3.9273 - acc: 0.1027 - val_loss: 3.9722 - val_acc: 0.0994
    Epoch 9/10
    6660/6680 [============================>.] - ETA: 0s - loss: 3.7479 - acc: 0.1308Epoch 00009: val_loss improved from 3.97221 to 3.94785, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 51s 8ms/step - loss: 3.7498 - acc: 0.1308 - val_loss: 3.9478 - val_acc: 0.1090
    Epoch 10/10
    6660/6680 [============================>.] - ETA: 0s - loss: 3.5748 - acc: 0.1583Epoch 00010: val_loss improved from 3.94785 to 3.75764, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 51s 8ms/step - loss: 3.5734 - acc: 0.1588 - val_loss: 3.7576 - val_acc: 0.1305
    




    <keras.callbacks.History at 0x7fd017e71358>




```python
## 加载具有最好验证loss的模型

model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### 测试模型

在狗图像的测试数据集上试用你的模型。确保测试准确率大于1%。


```python
# 获取测试数据集中每一个图像所预测的狗品种的index
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 14.8325%
    

---
<a id='step4'></a>
## 步骤 4: 使用一个CNN来区分狗的品种


使用 迁移学习（Transfer Learning）的方法，能帮助我们在不损失准确率的情况下大大减少训练时间。在以下步骤中，你可以尝试使用迁移学习来训练你自己的CNN。


### 得到从图像中提取的特征向量（Bottleneck Features）


```python
bottleneck_features = np.load('/data/bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### 模型架构

该模型使用预训练的 VGG-16 模型作为固定的图像特征提取器，其中 VGG-16 最后一层卷积层的输出被直接输入到我们的模型。我们只需要添加一个全局平均池化层以及一个全连接层，其中全连接层使用 softmax 激活函数，对每一个狗的种类都包含一个节点。


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_2 ( (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________
    


```python
## 编译模型

VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```


```python
## 训练模型

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6480/6680 [============================>.] - ETA: 0s - loss: 12.7966 - acc: 0.1060Epoch 00001: val_loss improved from inf to 11.24646, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 294us/step - loss: 12.7341 - acc: 0.1090 - val_loss: 11.2465 - val_acc: 0.1796
    Epoch 2/20
    6640/6680 [============================>.] - ETA: 0s - loss: 10.4738 - acc: 0.2617Epoch 00002: val_loss improved from 11.24646 to 10.24095, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 245us/step - loss: 10.4674 - acc: 0.2624 - val_loss: 10.2409 - val_acc: 0.2731
    Epoch 3/20
    6580/6680 [============================>.] - ETA: 0s - loss: 9.8305 - acc: 0.3264Epoch 00003: val_loss improved from 10.24095 to 9.94029, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 264us/step - loss: 9.8197 - acc: 0.3271 - val_loss: 9.9403 - val_acc: 0.3114
    Epoch 4/20
    6600/6680 [============================>.] - ETA: 0s - loss: 9.4904 - acc: 0.3676Epoch 00004: val_loss improved from 9.94029 to 9.74834, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 274us/step - loss: 9.4897 - acc: 0.3678 - val_loss: 9.7483 - val_acc: 0.3138
    Epoch 5/20
    6500/6680 [============================>.] - ETA: 0s - loss: 9.2638 - acc: 0.3889Epoch 00005: val_loss improved from 9.74834 to 9.47484, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 273us/step - loss: 9.2598 - acc: 0.3888 - val_loss: 9.4748 - val_acc: 0.3473
    Epoch 6/20
    6520/6680 [============================>.] - ETA: 0s - loss: 9.0086 - acc: 0.4109Epoch 00006: val_loss improved from 9.47484 to 9.24107, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 273us/step - loss: 9.0195 - acc: 0.4105 - val_loss: 9.2411 - val_acc: 0.3593
    Epoch 7/20
    6520/6680 [============================>.] - ETA: 0s - loss: 8.8275 - acc: 0.4302Epoch 00007: val_loss improved from 9.24107 to 9.23215, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 273us/step - loss: 8.8288 - acc: 0.4296 - val_loss: 9.2321 - val_acc: 0.3629
    Epoch 8/20
    6640/6680 [============================>.] - ETA: 0s - loss: 8.7753 - acc: 0.4389Epoch 00008: val_loss improved from 9.23215 to 9.19125, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 270us/step - loss: 8.7750 - acc: 0.4389 - val_loss: 9.1912 - val_acc: 0.3796
    Epoch 9/20
    6520/6680 [============================>.] - ETA: 0s - loss: 8.7501 - acc: 0.4442Epoch 00009: val_loss improved from 9.19125 to 9.14901, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 273us/step - loss: 8.7471 - acc: 0.4445 - val_loss: 9.1490 - val_acc: 0.3749
    Epoch 10/20
    6540/6680 [============================>.] - ETA: 0s - loss: 8.7039 - acc: 0.4466Epoch 00010: val_loss improved from 9.14901 to 9.14016, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 277us/step - loss: 8.6981 - acc: 0.4470 - val_loss: 9.1402 - val_acc: 0.3784
    Epoch 11/20
    6500/6680 [============================>.] - ETA: 0s - loss: 8.5646 - acc: 0.4563Epoch 00011: val_loss improved from 9.14016 to 9.02551, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 277us/step - loss: 8.5592 - acc: 0.4567 - val_loss: 9.0255 - val_acc: 0.3844
    Epoch 12/20
    6540/6680 [============================>.] - ETA: 0s - loss: 8.4057 - acc: 0.4659Epoch 00012: val_loss improved from 9.02551 to 8.89266, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 246us/step - loss: 8.3823 - acc: 0.4675 - val_loss: 8.8927 - val_acc: 0.3844
    Epoch 13/20
    6440/6680 [===========================>..] - ETA: 0s - loss: 8.1956 - acc: 0.4716Epoch 00013: val_loss improved from 8.89266 to 8.64131, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 243us/step - loss: 8.1811 - acc: 0.4722 - val_loss: 8.6413 - val_acc: 0.3880
    Epoch 14/20
    6560/6680 [============================>.] - ETA: 0s - loss: 7.9488 - acc: 0.4905Epoch 00014: val_loss improved from 8.64131 to 8.49218, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 251us/step - loss: 7.9572 - acc: 0.4901 - val_loss: 8.4922 - val_acc: 0.4096
    Epoch 15/20
    6620/6680 [============================>.] - ETA: 0s - loss: 7.8854 - acc: 0.5003Epoch 00015: val_loss did not improve
    6680/6680 [==============================] - 2s 271us/step - loss: 7.8910 - acc: 0.4996 - val_loss: 8.5065 - val_acc: 0.4132
    Epoch 16/20
    6660/6680 [============================>.] - ETA: 0s - loss: 7.8767 - acc: 0.5044Epoch 00016: val_loss did not improve
    6680/6680 [==============================] - 2s 275us/step - loss: 7.8752 - acc: 0.5043 - val_loss: 8.5100 - val_acc: 0.4084
    Epoch 17/20
    6460/6680 [============================>.] - ETA: 0s - loss: 7.8705 - acc: 0.5060Epoch 00017: val_loss improved from 8.49218 to 8.37019, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 274us/step - loss: 7.8604 - acc: 0.5066 - val_loss: 8.3702 - val_acc: 0.4192
    Epoch 18/20
    6620/6680 [============================>.] - ETA: 0s - loss: 7.7552 - acc: 0.5116Epoch 00018: val_loss improved from 8.37019 to 8.33058, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 275us/step - loss: 7.7837 - acc: 0.5096 - val_loss: 8.3306 - val_acc: 0.4216
    Epoch 19/20
    6660/6680 [============================>.] - ETA: 0s - loss: 7.7250 - acc: 0.5125Epoch 00019: val_loss improved from 8.33058 to 8.24350, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 275us/step - loss: 7.7285 - acc: 0.5121 - val_loss: 8.2435 - val_acc: 0.4299
    Epoch 20/20
    6520/6680 [============================>.] - ETA: 0s - loss: 7.6838 - acc: 0.5140Epoch 00020: val_loss improved from 8.24350 to 8.18229, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 271us/step - loss: 7.6827 - acc: 0.5141 - val_loss: 8.1823 - val_acc: 0.4335
    




    <keras.callbacks.History at 0x7fd03013d908>




```python
## 加载具有最好验证loss的模型

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### 测试模型
现在，我们可以测试此CNN在狗图像测试数据集中识别品种的效果如何。我们在下方打印出测试准确率。


```python
# 获取测试数据集中每一个图像所预测的狗品种的index
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 42.2249%
    

### 使用模型预测狗的品种


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # 提取bottleneck特征
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## 步骤 5: 建立一个CNN来分类狗的品种（使用迁移学习）

现在你将使用迁移学习来建立一个CNN，从而可以从图像中识别狗的品种。你的 CNN 在测试集上的准确率必须至少达到60%。

在步骤4中，我们使用了迁移学习来创建一个使用基于 VGG-16 提取的特征向量来搭建一个 CNN。在本部分内容中，你必须使用另一个预训练模型来搭建一个 CNN。为了让这个任务更易实现，我们已经预先对目前 keras 中可用的几种网络进行了预训练：

- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

这些文件被命名为为：

    Dog{network}Data.npz

其中 `{network}` 可以是 `VGG19`、`Resnet50`、`InceptionV3` 或 `Xception` 中的一个。选择上方网络架构中的一个，他们已经保存在目录 `/data/bottleneck_features/` 中。


### 【练习】获取模型的特征向量

在下方代码块中，通过运行下方代码提取训练、测试与验证集相对应的bottleneck特征。

    bottleneck_features = np.load('/data/bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: 从另一个预训练的CNN获取bottleneck特征
bottleneck_features = np.load('/data/bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']
```

### 【练习】模型架构

建立一个CNN来分类狗品种。在你的代码单元块的最后，通过运行如下代码输出网络的结构：
    
        <your model's name>.summary()
   
---

<a id='question6'></a>  

### __问题 6:__ 


在下方的代码块中尝试使用 Keras 搭建最终的网络架构，并回答你实现最终 CNN 架构的步骤与每一步的作用，并描述你在迁移学习过程中，使用该网络架构的原因。


__回答:__ 首先,声明一个Sequential,深度学习的前几层是将分析图片的线条,角度,简单图形等图片的基础信息,所以使用迁移学习是将图片输入Resnet50中在最后一层停止学习,然后将其导出,之后将结果导入自己的框架,由于传入模型的图片已经经过了Resnet50,所以是一个宽高为1,1维度为2048的一个矩阵,第一步将这个矩阵通过Flatten进行扁平化处理,随后通过全连接层进行学习,狗的种类有133种,同时以softmax为误差函数使得输出一个0-1之间的数字,方便进行判断

四个迁移模型都是对imageNet中的图片进行分类后训练得到的,说以均适用于分类任务,对几个框架进行测试后发现对Resnet50框架进行迁移的效果最好,所以使用Resnet50

第三步中简单CNN的层数较低,模型相对简单,难以得到极好的效果

而第四步中使用的VGG16层数也相对较少,难以达到更好的效果

vgg模型结构简答有效,前几层仅用3 * 3 卷积核来增加网络深度,通过最大池化层依次减少每层的神经元数量

Resnet加入了恒等映射层让网格在深度增加的情况下不退化,使用恒等映射来更新残差模块,获得很高的准确性,且模型本身很小

Inception模块的目的是充当"多级特征提取器",采用1 * 1 , 3 * 3 , 5 * 5 的卷积核最后把卷积核输出链接当做下一层,本身更小小于VGG和ResNet

Xception的weight数量最少,只有91M


```python
### TODO: 定义你的框架
model = Sequential()
model.add(Flatten(input_shape=(1, 1, 2048)))
model.add(Dense(133, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```


```python
### TODO: 编译模型
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_2 (Flatten)          (None, 2048)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 133)               272517    
    =================================================================
    Total params: 272,517
    Trainable params: 272,517
    Non-trainable params: 0
    _________________________________________________________________
    

---

### 【练习】训练模型

<a id='question7'></a>  

### __问题 7:__ 

在下方代码单元中训练你的模型。使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。

当然，你也可以对训练集进行 [数据增强](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 以优化模型的表现，不过这不是必须的步骤。



```python
### TODO: 训练模型
checkpointer = ModelCheckpoint(filepath='dogResnet50.weights.best.hdf5', verbose=1, save_best_only=True)
model.fit(train_Resnet50, train_targets, epochs=20, validation_data=(valid_Resnet50, valid_targets), 
          callbacks=[checkpointer], verbose=1, shuffle=True)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6432/6680 [===========================>..] - ETA: 0s - loss: 1.7977 - acc: 0.5734Epoch 00001: val_loss improved from inf to 0.83568, saving model to dogResnet50.weights.best.hdf5
    6680/6680 [==============================] - 2s 225us/step - loss: 1.7633 - acc: 0.5790 - val_loss: 0.8357 - val_acc: 0.7521
    Epoch 2/20
    6624/6680 [============================>.] - ETA: 0s - loss: 0.4593 - acc: 0.8590Epoch 00002: val_loss improved from 0.83568 to 0.69910, saving model to dogResnet50.weights.best.hdf5
    6680/6680 [==============================] - 1s 171us/step - loss: 0.4605 - acc: 0.8587 - val_loss: 0.6991 - val_acc: 0.7796
    Epoch 3/20
    6656/6680 [============================>.] - ETA: 0s - loss: 0.2625 - acc: 0.9205Epoch 00003: val_loss did not improve
    6680/6680 [==============================] - 1s 153us/step - loss: 0.2619 - acc: 0.9208 - val_loss: 0.7016 - val_acc: 0.7940
    Epoch 4/20
    6336/6680 [===========================>..] - ETA: 0s - loss: 0.1682 - acc: 0.9511Epoch 00004: val_loss did not improve
    6680/6680 [==============================] - 1s 152us/step - loss: 0.1679 - acc: 0.9512 - val_loss: 0.7125 - val_acc: 0.7737
    Epoch 5/20
    6432/6680 [===========================>..] - ETA: 0s - loss: 0.1118 - acc: 0.9656Epoch 00005: val_loss improved from 0.69910 to 0.67172, saving model to dogResnet50.weights.best.hdf5
    6680/6680 [==============================] - 1s 153us/step - loss: 0.1118 - acc: 0.9654 - val_loss: 0.6717 - val_acc: 0.8096
    Epoch 6/20
    6368/6680 [===========================>..] - ETA: 0s - loss: 0.0748 - acc: 0.9794Epoch 00006: val_loss did not improve
    6680/6680 [==============================] - 1s 153us/step - loss: 0.0751 - acc: 0.9798 - val_loss: 0.6802 - val_acc: 0.8263
    Epoch 7/20
    6656/6680 [============================>.] - ETA: 0s - loss: 0.0509 - acc: 0.9878Epoch 00007: val_loss did not improve
    6680/6680 [==============================] - 1s 171us/step - loss: 0.0518 - acc: 0.9876 - val_loss: 0.6728 - val_acc: 0.8216
    Epoch 8/20
    6432/6680 [===========================>..] - ETA: 0s - loss: 0.0366 - acc: 0.9911Epoch 00008: val_loss improved from 0.67172 to 0.63540, saving model to dogResnet50.weights.best.hdf5
    6680/6680 [==============================] - 1s 177us/step - loss: 0.0369 - acc: 0.9910 - val_loss: 0.6354 - val_acc: 0.8275
    Epoch 9/20
    6432/6680 [===========================>..] - ETA: 0s - loss: 0.0276 - acc: 0.9946Epoch 00009: val_loss did not improve
    6680/6680 [==============================] - 1s 176us/step - loss: 0.0284 - acc: 0.9943 - val_loss: 0.6467 - val_acc: 0.8311
    Epoch 10/20
    6464/6680 [============================>.] - ETA: 0s - loss: 0.0197 - acc: 0.9960Epoch 00010: val_loss did not improve
    6680/6680 [==============================] - 1s 173us/step - loss: 0.0199 - acc: 0.9958 - val_loss: 0.7016 - val_acc: 0.8263
    Epoch 11/20
    6432/6680 [===========================>..] - ETA: 0s - loss: 0.0159 - acc: 0.9966Epoch 00011: val_loss did not improve
    6680/6680 [==============================] - 1s 175us/step - loss: 0.0158 - acc: 0.9966 - val_loss: 0.6887 - val_acc: 0.8287
    Epoch 12/20
    6432/6680 [===========================>..] - ETA: 0s - loss: 0.0125 - acc: 0.9978Epoch 00012: val_loss did not improve
    6680/6680 [==============================] - 1s 174us/step - loss: 0.0127 - acc: 0.9976 - val_loss: 0.7121 - val_acc: 0.8251
    Epoch 13/20
    6496/6680 [============================>.] - ETA: 0s - loss: 0.0099 - acc: 0.9977Epoch 00013: val_loss did not improve
    6680/6680 [==============================] - 1s 173us/step - loss: 0.0101 - acc: 0.9978 - val_loss: 0.7731 - val_acc: 0.8192
    Epoch 14/20
    6464/6680 [============================>.] - ETA: 0s - loss: 0.0087 - acc: 0.9980Epoch 00014: val_loss did not improve
    6680/6680 [==============================] - 1s 175us/step - loss: 0.0087 - acc: 0.9981 - val_loss: 0.7555 - val_acc: 0.8347
    Epoch 15/20
    6432/6680 [===========================>..] - ETA: 0s - loss: 0.0078 - acc: 0.9981Epoch 00015: val_loss did not improve
    6680/6680 [==============================] - 1s 174us/step - loss: 0.0079 - acc: 0.9981 - val_loss: 0.7321 - val_acc: 0.8228
    Epoch 16/20
    6592/6680 [============================>.] - ETA: 0s - loss: 0.0052 - acc: 0.9986Epoch 00016: val_loss did not improve
    6680/6680 [==============================] - 1s 172us/step - loss: 0.0059 - acc: 0.9985 - val_loss: 0.7648 - val_acc: 0.8263
    Epoch 17/20
    6496/6680 [============================>.] - ETA: 0s - loss: 0.0052 - acc: 0.9980Epoch 00017: val_loss did not improve
    6680/6680 [==============================] - 1s 175us/step - loss: 0.0051 - acc: 0.9981 - val_loss: 0.8232 - val_acc: 0.8251
    Epoch 18/20
    6400/6680 [===========================>..] - ETA: 0s - loss: 0.0050 - acc: 0.9986Epoch 00018: val_loss did not improve
    6680/6680 [==============================] - 1s 176us/step - loss: 0.0049 - acc: 0.9987 - val_loss: 0.8564 - val_acc: 0.8371
    Epoch 19/20
    6432/6680 [===========================>..] - ETA: 0s - loss: 0.0052 - acc: 0.9988Epoch 00019: val_loss did not improve
    6680/6680 [==============================] - 1s 177us/step - loss: 0.0051 - acc: 0.9988 - val_loss: 0.8159 - val_acc: 0.8479
    Epoch 20/20
    6624/6680 [============================>.] - ETA: 0s - loss: 0.0044 - acc: 0.9988Epoch 00020: val_loss did not improve
    6680/6680 [==============================] - 1s 169us/step - loss: 0.0044 - acc: 0.9988 - val_loss: 0.8927 - val_acc: 0.8263
    




    <keras.callbacks.History at 0x7fd017cc4a20>




```python
### TODO: 加载具有最佳验证loss的模型权重
model.load_weights('dogResnet50.weights.best.hdf5')
```

---

### 【练习】测试模型

<a id='question8'></a>  

### __问题 8:__ 

在狗图像的测试数据集上试用你的模型。确保测试准确率大于60%。


```python
### TODO: 在测试集上计算分类准确率
# 获取测试数据集中每一个图像所预测的狗品种的index
Resnet50_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 82.6555%
    

---

### 【练习】使用模型测试狗的品种


实现一个函数，它的输入为图像路径，功能为预测对应图像的类别，输出为你模型预测出的狗类别（`Affenpinscher`, `Afghan_hound` 等）。

与步骤5中的模拟函数类似，你的函数应当包含如下三个步骤：

1. 根据选定的模型载入图像特征（bottleneck features）
2. 将图像特征输输入到你的模型中，并返回预测向量。注意，在该向量上使用 argmax 函数可以返回狗种类的序号。
3. 使用在步骤0中定义的 `dog_names` 数组来返回对应的狗种类名称。

提取图像特征过程中使用到的函数可以在 `extract_bottleneck_features.py` 中找到。同时，他们应已在之前的代码块中被导入。根据你选定的 CNN 网络，你可以使用 `extract_{network}` 函数来获得对应的图像特征，其中 `{network}` 代表 `VGG19`, `Resnet50`, `InceptionV3`, 或 `Xception` 中的一个。
 
---

<a id='question9'></a>  

### __问题 9:__


```python
### TODO: 写一个函数，该函数将图像的路径作为输入
### 然后返回此模型所预测的狗的品种
def name_of_dog(path):
    characteristic = extract_Resnet50(path_to_tensor(path))
    lis = model.predict(characteristic)
    return dog_names[np.argmax(lis)]
    
print(name_of_dog(human_files[2]))
```

    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    94658560/94653016 [==============================] - 6s 0us/step
    in/127.Silky_terrier
    

---

<a id='step6'></a>
## 步骤 6: 完成你的算法



实现一个算法，它的输入为图像的路径，它能够区分图像是否包含一个人、狗或两者都不包含，然后：

- 如果从图像中检测到一只__狗__，返回被预测的品种。
- 如果从图像中检测到__人__，返回最相像的狗品种。
- 如果两者都不能在图像中检测到，输出错误提示。

我们非常欢迎你来自己编写检测图像中人类与狗的函数，你可以随意地使用上方完成的 `face_detector` 和 `dog_detector` 函数。你__需要__在步骤5使用你的CNN来预测狗品种。

下面提供了算法的示例输出，但你可以自由地设计自己的模型！

![Sample Human Output](images/sample_human_output.png)




<a id='question10'></a>  

### __问题 10:__

在下方代码块中完成你的代码。

---



```python
### TODO: 设计你的算法
### 自由地使用所需的代码单元数吧
def human_or_dog(path):
    if face_detector(path):
        return "this dog like a ...{0}".format(name_of_dog(path))
    elif dog_detector(path):
        return "you look like a ...{0}".format(name_of_dog(path))
    else:
        return "error"

print(human_or_dog(human_files[23]))
```

    this dog like a ...in/064.English_toy_spaniel
    

---
<a id='step7'></a>
## 步骤 7: 测试你的算法

在这个部分中，你将尝试一下你的新算法！算法认为__你__看起来像什么类型的狗？如果你有一只狗，它可以准确地预测你的狗的品种吗？如果你有一只猫，它会将你的猫误判为一只狗吗？


<a id='question11'></a>  

### __问题 11:__

在下方编写代码，用至少6张现实中的图片来测试你的算法。你可以使用任意照片，不过请至少使用两张人类图片（要征得当事人同意哦）和两张狗的图片。
同时请回答如下问题：

1. 输出结果比你预想的要好吗 :) ？或者更糟 :( ？
2. 提出至少三点改进你的模型的想法。

结果和预想的结果相似

模型的改进建议:
1. 前有不同的模型所得结果并不相同,在条件允许的情况下可以将使用不同的迁移模型做成流水线,选择最适合这个项目的迁移模型
2. 在项目中加入ImageDataGenerator模块强化他的图片不变性,同时防止过拟合
3. 使用流水线或者手动调整模型中的参数如卷积核大小等,得到并使用最佳参数,来使模型得到更加优秀的效果



```python
## TODO: 在你的电脑上，在步骤6中，至少在6张图片上运行你的算法。
## 自由地使用所需的代码单元数吧
print(human_or_dog('zxy.jpg'))
```

    this dog like a ...in/016.Beagle
    

**注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出File -> Download as -> HTML (.html)把这个 HTML 和这个 iPython notebook 一起做为你的作业提交。**


```python
print(human_or_dog('ljh.jpg'))
```

    this dog like a ...in/101.Maltese
    


```python
print(human_or_dog('cz.jpg'))
```

    this dog like a ...in/048.Chihuahua
    


```python
print(human_or_dog('qt.jpg'))
```

    you look like a ...in/004.Akita
    


```python
print(human_or_dog('zt.jpg'))
```

    error
    


```python
print(human_or_dog('cq.jpg'))
```

    you look like a ...in/053.Cocker_spaniel
    
