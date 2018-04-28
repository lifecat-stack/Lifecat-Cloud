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
### 1、主程序
```python
def run():  
    """ 
    主程序 
    """  
    img = cv2.imread("IMGP8080.JPG")  
    arrimg = np.array(img)  
    # 获取像素点的集合  
    feature = loadDataSet(arrimg)  
    # 获取初始聚类中心  
    init_cen_1, init_cen_2 = sel_init_cen(feature)  
    # 三种方式获得聚类中心  
    n = input("请选择function:\n"  
              "1-->第五题窗口法\n"  
              "2-->第六题kmeans聚类法\n"  
              "3-->快速验证(使用计算好的聚类中心)\n")  
    cen1, cen2 = sel_function(n, feature, arrimg, init_cen_1, init_cen_2)  
    # 打印结果  
    print("最终结果:", cen1, cen2)  
    # 图片分类及展示图片  
    image2k(cen1, cen2)  
```
### 2、读取图片数据到数组  
```python
def loadDataSet(arrimg):  
    """ 
    读取图片数据(三维向量b,g,r) 到 feature(一维列表[r,g,b]) 数据集 
    :param arrimg: jpg图片的numpy数组形式 
    :return: 特征向量组feature 
    """  
    row = arrimg.shape[0]  
    col = arrimg.shape[1]  
  
    # print("数组行数:",row)  
    # print("数组列数:",col)  
  
    # 特征向量集合  
    features = []  
  
    print("正在读取图片信息，请稍等......")  
  
    # (行,列):读取像素点rgb数据  
    for i in range(0, row):  
        for j in range(0, col):  
            r = arrimg[i, j, 2]  
            g = arrimg[i, j, 1]  
            b = arrimg[i, j, 0]  
            features.append([r, g, b])  
  
    # 转换成numpy矩阵计算形式，浮点型f  
    feature = np.array(features, 'f')  
  
    # print(feature,"以上为像素点集合")  
    # print("像素点集合的shape:", feature.shape)  
  
    return feature  
```
### 3、判断“距离”
```python
def distance(vecA, vecB):  
    """ 
    计算两个向量的距离:三维向量-->一维距离 
    :param vecA: 向量A 
    :param vecB: 向量B 
    :return: 浮点型数据 
    """  
    return np.sqrt(np.power(vecA[0] - vecB[0], 2) + np.power(vecA[1] - vecB[1], 2) + np.power(vecA[2] - vecB[2], 2))  
```
### PS：选取不同的方法进行聚类
```python
def sel_function(n, feature, arrimg, init_cen_1, init_cen_2):  
    """ 
    三选一，获得聚类中心 
    :param n: 哪种方式？1,2,3 
    :param feature: 特征向量数据集 
    :param arrimg: 图片数据集 
    :param init_cen_1: 初始聚类中心 
    :param init_cen_2: 初始聚类中心 
    :return: 聚类中心 cen1 cen2 
    """  
    if n == 1:  
        # 第1种方法 window获得聚类中心  
        cen_1, cen_2 = window_get_centor(arrimg)  
    elif n == 2:  
        # 第2种方法 kmeans获得聚类中心  
        cen_1, cen_2 = kmeans_get_centor(feature, init_cen_1, init_cen_2)  
    elif n == 3:  
        # 第3种方法直接获得计算结果，节省时间  
        cen_1 = [95.81982617, 69.23093989, 57.99697231]  
        cen_2 = [207.61375807, 152.36148107, 124.72904938]  
  
    else:  
        cen_1 = init_cen_1  
        cen_2 = init_cen_2  
        print("无此方法，请输入1,2,3")  
  
    return cen_1, cen_2  
```
### 4、选取kmeans初始聚类中心
```python
def sel_init_cen(features):  
    """ 
    随机选择两个初始点 
    :param features: 图像的特征向量集,[[r,g,b],[r,g,b]...] 
    :return: 初始聚类中心 cen1 cen2 
    """  
    # 选取随机数  
    rand_1 = (int)(np.random.random() * (features.shape[0]))  
    rand_2 = (int)(np.random.random() * (features.shape[0]))  
  
    # 选取初始中心  
    centor_1 = features[rand_1]  
    centor_2 = features[rand_2]  
  
    print("初始聚类中心为:", centor_1, "+", centor_2)  
  
    return centor_1, centor_2  
```
### 5、kmeans进行聚类
```python
def kmeans(node, centor1, centor2, class1, class2):  
    """ 
    kmeans方法迭代计算聚类中心 
    :param node: 待判断数据[r,g,b] 
    :param centor1: 当前聚类中心 
    :param centor2: 当前聚类中心 
    :param class1: 属于类1的数据集 
    :param class2: 属于类2的数据集 
    :return: 新的聚类中心 cen1 cen2 
    """  
    dist1 = distance(node, centor1)  
    dist2 = distance(node, centor2)  
  
    # 判断新数据和两个聚类中心的距离  
    if dist1 < dist2:  
        print(node, "添加到class1")  
        class1.append(node)  
        centor1 = np.mean(class1, axis=0)  
  
    elif dist2 < dist1:  
        class2.append(node)  
        print(node, "添加到class2")  
        centor2 = np.mean(class2, axis=0)  
  
    else:  
        class1.append(node)  
        class2.append(node)  
        print(node, "添加到c1和c2")  
        centor1 = np.mean(class1, axis=0)  
        centor2 = np.mean(class2, axis=0)  
  
    print("mean-class1", centor1)  
    print("mean-class2", centor2)  
  
    return centor1, centor2  
```
### PS：窗口法进行分类：
```python
def window_get_centor(arrimg):  
    """ 
    第1种方式：window-->fisher判别方法获得聚类中心 
    1、获得数据集类1，类2 
    2、计算均值向量 mean1 mean2 
    3、计算类内离散度矩阵 sw_1 sw_2 
    4、计算总类内离散度矩阵 sw=sw_1+sw_2 
    5、获得sw的逆矩阵 sw.I 
    6、计算w= 
 
    :param arrimg: 图片数据集 
    :return: 结果聚类中心 cen1 cen2 
    """  
    # 图片长宽  
    length = arrimg.shape[1]  
    width = arrimg.shape[0]  
    # 类1 的窗口  
    cl1_x_str = (int)(0.44 * length)  
    cl1_x_end = (int)(0.55 * length)  
    cl1_y_str = (int)(0.44 * width)  
    cl1_y_end = (int)(0.55 * width)  
    # 类2 的窗口  
    cl2_x_str = (int)(0.77 * length)  
    cl2_x_end = (int)(0.88 * length)  
    cl2_y_str = (int)(0.44 * width)  
    cl2_y_end = (int)(0.55 * width)  
  
    class1 = []  
    class2 = []  
  
    # 获取类1  
    for row1 in range(cl1_x_str, cl1_x_end):  
        for col1 in range(cl1_y_str, cl1_y_end):  
            class1.append(arrimg[col1][row1])  
    # 获取类2  
    for row2 in range(cl2_x_str, cl2_x_end):  
        for col2 in range(cl2_y_str, cl2_y_end):  
            class2.append(arrimg[col2][row2])  
  
    feature1 = np.array(class1, 'f')  
    feature2 = np.array(class2, 'f')  
    cen_1 = np.mean(feature1, axis=0)  
    cen_2 = np.mean(feature2, axis=0)  
  
    print("window计算类1中心为：", cen_1)  
    print("window计算类2中心为：", cen_2)  
  
    return cen_1, cen_2  
```
### 6、获取聚类中心
```python
def kmeans_get_centor(feature, cen_1, cen_2):  
    """ 
    第2种方式：kmeans方法获得聚类中心 
    :param feature: 数据集 
    :param cen_1: 初始聚类中心 
    :param cen_2: 初始聚类中心 
    :return: 结果聚类中心 cen1 cen2 
    """  
    # 建立两个聚类的空集合，注意：引入更正数据[0,0,0]，防止数据过大  
    class1 = []  
    class2 = []  
    class1.append([0, 0, 0])  
    class2.append([0, 0, 0])  
  
    # 第一种方式：设置大步长，减少计算时间  
    for i in range(0, feature.shape[0] - 1, 100):  
        print("当前为第", i, "轮")  
        # 使用kmeans计算方法  
        centor1, centor2 = kmeans(feature[i], cen_1, cen_2, class1, class2)  
        cen_1 = centor1  
        cen_2 = centor2  
  
        # 第二种方式：计算两次迭代之间的均方差---->当均方差<0.0001时，停止迭代  
        pass  
  
    return cen_1, cen_2  
```
### 7、显示聚类图像
```python
def image2k(centor1, centor2):  
    """ 
    根据聚类中心进行图像二分类 
    :param centor1: 结果聚类中心 
    :param centor2: 结果聚类中心 
    :return: 显示图像 
    """  
    img2 = cv2.imread("IMGP8080.JPG")  
    print(img2.shape)  
    row = img2.shape[0]  
    col = img2.shape[1]  
  
    # (行,列):设置像素点bgr数据  
    for i in range(0, row):  
        for j in range(0, col):  
            print("图像分类已进行到: 第", i, "行+", j, "列")  
            # 类1 黑色 类型[b,g,r]  
            if distance(img2[i][j], centor1) < distance(img2[i][j], centor2):  
                img2[i][j] = [0, 0, 0]  
            # 类2 红色 类型[b,g,r]  
            else:  
                img2[i][j] = [0, 0, 255]  
  
    # 窗口,调整图像大小  
    win = cv2.namedWindow('img win', flags=0)  
    cv2.imshow('img win', img2)  
    cv2.waitKey(0)  
```
