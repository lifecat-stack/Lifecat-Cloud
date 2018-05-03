# Python-Machine-Learn
人工智能-机器学习算法笔记
------
所有算法均为基本数据结构实现，未调用相关库
------
# bayes:  

   对于Iris数据库，假设样本数据服从多变量正态分布。用Python实现以下内容：  
   1）用最大似然估计方法估计先验概率密度。
   2）建立最小错误率Bayes分类器。
   3）检验分类器性能。

### 数学推导

* （1）最大似然估计方法估计先验概率：

设样本类型为Y，样本有Ck种类别，k=1,2,3   
极大似然估计先验概率为：  
P（Y=Ck）=∑（i=1,2,…N）I（yi=ck）/N，k=1,2,3
		
∵指示函数I（yi=ck）=1（当yi=ck成立）或0（当yi=ck不成立）  
∴先验概率P（Y=Ck）= 样本中Ck的数目/样本总数，k=1,2,3  

∴P（Y='Iris-setosa'）=∑I（Y=C1）/N总  
  P（Y='Iris-versicolor'）=∑I（Y=C2）/N总  
  P（Y='Iris-virginica'）=∑I（Y=C3）/N总  

* （2）建立最小错误率Bayes分类器  

已知输入变量X={x1,x2,x3,x4}服从四维正态分布 N4（μ，∑），μ为样本均值，∑为样本协方差矩阵  
四维正态分布概率密度函数如下：  
f（X|μ，∑）=[1/gen(2π)^d * gen(|∑|)]exp[-1/2 * (X-μ)∑^-1(X-μ)T]  

已知先验概率P（Y=Ck）= 1/3  k=1,2,3  

求后验概率为P（Y=Ck|X）= P（X|Y=Ck）P（Y=Ck）/∑P（X|Y=Ck）P（Y=Ck）  

∴需要根据样本数据估计μ，∑  
∑=[cov(x1,x1),…,…,…]		u1=x1   μ=（μ1，μ2，μ3，μ4）T  
   […,cov(x2,x2),…,…]		u2=x2   
   […,…,cov(x3,x3),…]		u3=x3  
   […,…,…,cov(x4,x4)]		u4=x4  

得到条件概率密度P（X|Y=Ck），k=1,2,3  
从而得到后验概率P（Y=Ck|X）,k=1,2,3  

又因为在三个类别中先验概率密度P（Y=Ck）相等，分母∑P（X|Y=Ck）P（Y=Ck）相等  
故只需判断条件概率密度P（X|Y=Ck）  

∴通过判别一个输入X的 MAX=max{P（X|Y=C1）, P（X|Y=C2）, P（X|Y=C3）}  
若 MAX=P（X|Y=C1）,则X为C1类  
若 MAX=P（X|Y=C2）,则X为C2类  
若 MAX=P（X|Y=C3）,则X为C3类  
  
* （3）检验分类器性能  

采用0-1损失函数，对数据集进行K折交叉验证，划分为K个不相交的子集，  
每次选取K-1个子集进行训练，其余1个子集进行验证，得到测试误差Etest。  
重复K次，计算Etest的均值，即为正确率。  

------
# kmeans：
* kmeans算法未调用库，使用基本数据结构实现  

  1. 对于给定的图片IMGP8080.jpg，要求把河流部分划分出来。可以采用以下方法：在该图像中分别在河流部分与非河流部分画出一个窗口，把在这两个窗口中的像素数据作为训练集，用Fisher线性判别方法求得分类器参数，再用该分类器对整幅图进行分类。请用python程序实现。

  2. 对于给定的图片IMGP8080.jpg，要求利用聚类方法把图片分成2类，不要求标定。具体聚类算法不限，在实验报告中叙述清楚即可。请用python程序实现。

注：可以使用OpenCV或者其他任何图像处理的框架，但要求自己完成分类或者聚类算法。



CSDN教程：https://blog.csdn.net/wsh596823919/article/details/79981703



实验效果图：  
![image](https://img-blog.csdn.net/20180417221403327?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  
![image](https://img-blog.csdn.net/20180417221451403?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dzaDU5NjgyMzkxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)  



python算法实现：

```python
import sys
import numpy as np
import cv2


def loadDataSet(arrimg):
    """
    读取numpy.array()图像数据

        from: array([[b,g,r],[b,g,r]...],
                    ....................
                    [[b,g,r],[b,g,r]...])

        to: [[r,g,b],[r,g,b]...]

    :param arrimg: array([[b,g,r],[b,g,r]...],[[],[]],......)
    :return: features=[[r,g,b],[r,g,b]...]
    """
    print("正在读取图片信息，请稍等......")

    row = arrimg.shape[0]
    col = arrimg.shape[1]

    features = []

    # read [r,g,b]
    for i in range(0, row):
        for j in range(0, col):
            r = arrimg[i, j, 2]
            g = arrimg[i, j, 1]
            b = arrimg[i, j, 0]
            features.append([r, g, b])

    features = np.array(features, 'f')
    return features


def distance(vecA, vecB):
    """
    计算rgb向量的欧式距离

    :param vecA: valueof[r,g,b]
    :param vecB: valueof[r.g.b]
    :return: dist
    """
    return np.sqrt(np.power(vecA[0] - vecB[0], 2) + np.power(vecA[1] - vecB[1], 2) + np.power(vecA[2] - vecB[2], 2))


def sel_init_cen(features, k):
    """
    随机选择K个初始聚类中心

    :param features: [[r,g,b],[r,g,b]...]
    :return: centors=[cen1,cen2...]
    """
    # 选取随机数
    rands = [(int)(np.random.random() * (features.shape[0])) for _ in range(k)]
    # 选取初始中心
    centors = [features[rands[i]] for i in range(k)]
    return centors


def get_centor(feature, centors):
    """
    迭代计算聚类中心

    :param node: 待判断数据[r,g,b]
    :param centors: init[cen1,cen2...]
    :param classes: [[node of class1],[node of class2],......[node of classk]]
    :return: cens=[cen1,cen2...]
    """
    k = len(centors)

    # 建立k个类别数据的空集合
    classes = [[] for _ in range(k)]

    # 设置大步长，减少计算时间
    for i in range(0, feature.shape[0] - 1, 100):

        # node到k个聚类中心的距离
        dists = [distance(feature[i], centor) for centor in centors]

        # 判为距离最近的类别，并重新计算聚类中心(平均值)
        for j in range(k):
            if min(dists) == distance(feature[i], centors[j]):
                classes[j].append(feature[i])
                centors[j] = np.mean(classes[j], axis=0)
                break

    return centors


def image2k(imagepath, centors):
    """
    根据聚类中心进行图像分类

    :param centors: 聚类中心
    :return: 显示图像
    """
    img2 = cv2.imread(imagepath)
    row = img2.shape[0]
    col = img2.shape[1]
    k = len(centors)

    # 定义颜色库 8
    colors = [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255],
              [255, 255, 0], [0, 255, 255], [255, 0, 255]]

    # (行,列):根据类别设置像素bgr数据
    for i in range(0, row):
        for j in range(0, col):
            print("图像分类已进行到:", i+1, "/", row, "行", j+1, "/", col, "列")
            # 当前像素到k个聚类中心的距离
            dists = [distance(img2[i][j], centor) for centor in centors]
            for ks in range(k):
                if min(dists) == distance(img2[i][j], centors[ks]):
                    img2[i][j] = colors[ks % len(colors)]

    # 窗口,调整图像大小
    win = cv2.namedWindow('kmeans', flags=0)
    cv2.imshow('kmeans', img2)
    cv2.waitKey(0)


def main(imagepath, k):
    """
    主程序
    """
    if k < 2 | k > 9:
        print('k is error')
        sys.exit(0)
    # numpy获取图像bgr数组
    arrimg = np.array(cv2.imread(imagepath))
    # 获取[[r,g,b]...]
    feature = loadDataSet(arrimg)
    # 获取k个随机初始聚类中心
    init_cens = sel_init_cen(feature, k)
    # 计算k个聚类中心
    cens = get_centor(feature, init_cens)
    # 显示k分类的图像
    image2k(imagepath, cens)


if __name__ == '__main__':
    # 获取图片路径以及k聚类数
    main('01.tiff', 4)

```
