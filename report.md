#  Project4：空域滤波器

姓名：江朝昀

班级：自动化少61

学号：2140506069

提交日期：2019.3.19

##  摘要

使用低通滤波器对图像进行平滑处理，以降低噪声。使用的低通滤波器有中值滤波器和高斯滤波器，其中要求设计的高斯滤波器的方差$\sigma=1.5$。然后用高斯低通滤波器得出的结果通过unsharp masking处理进行图像锐化，以及通过Sobel边算子、Laplace算子和Canny algorithm进行边缘检测。

##  一. 空域低通滤波器

题目要求：分别用高斯滤波器和中值滤波器去平滑测试图像test1和2，模板大小分别是3x3 ， 5x5 ，7x7。分析各自优缺点。其中，利用固定方差 sigma=1.5产生高斯滤波器。

###  1. 高斯滤波器

高斯平滑是邻域平均的思想对图像进行平滑的一种方法。在图像高斯平滑中，对图像进行平均时，不同位置的像素被赋予了不同的权重。高斯滤波器的模板是通过高斯函数计算出来的，以5*5的模板举例，首先在模板上建立一个坐标系，其原点是高斯模板的中心：

(-2, 2) (-1, 2) ( 0, 2) ( 1, 2) ( 2, 2)

(-2, 1) (-1, 1) ( 0, 1) ( 1, 1) ( 2, 1)

(-2, 0) (-1, 0) ( 0, 0) ( 1, 0) ( 2, 0)

(-2,-1) (-1,-1) ( 0,-1) ( 1,-1) ( 2,-1)

(-2,-2) (-1,-2) ( 0,-2) ( 1,-2) ( 2,-2)

每一个位置的权重值由公式计算得出：

![](https://latex.codecogs.com/gif.latex?G%28x%2Cy%29%3D%5Cfrac%7B1%7D%7B2%5CPi%5Csigma%5E%7B2%7D%7De%5E%7B-%5Cfrac%7Bx%5E%7B2%7D&plus;y%5E%7B2%7D%7D%7B2%5Csigma%5E%7B2%7D%7D%7D)

其中σ是方差，计算出结果后再进行归一化就可以对源图像做平滑处理了。实现高斯滤波器模板的程序如下：

```python
def creategaussian(size1, std1):
    ss = (size1 - 1) // 2
    sum_g = 0
    g = np.zeros((size1, size1))
    for ii in range(size1):
        for jj in range(size1):
            arg = -(pow((-ss + ii), 2) + pow((-ss + jj), 2)) / (2 * pow(std1, 2))
            g[ii][jj] = math.exp(arg)
            sum_g += g[ii][jj]
    for ii in range(size1):
        for jj in range(size1):
            g[ii][jj] = g[ii][jj] / sum_g
    return g
```

这里还涉及一个问题，就是用高斯模板去乘源图像时，当边缘像素为中心点时周围会有缺少、不足的位置，所以开始之前要将源图像拓展边缘，OpenCV库中有函数可以直接完成：

```python
arr = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT, value=0)
```

其中s代表上下左右要拓展的数量，对于灰度图像直接用0来填充。

第一次处理之后，处理的结果是类似于雪花屏的错误图像，我分析的原因是，我直接在原图像上进行修改了，这样处理后的效果会一直叠加，导致错误。

下面是分别使用3×3、5×5和7×7大小的高斯滤波器平滑test1和test2的结果。

![test1-gaussian-3](<https://github.com/jzy124/hw4/raw/master/pictures/1/test1-gaussian-3.png>)

![test1-gaussian-5](<https://github.com/jzy124/hw4/raw/master/pictures/1/test1-gaussian-5.png>)

![test1-gaussian-7](<https://github.com/jzy124/hw4/raw/master/pictures/1/test1-gaussian-7.png>)

![test2-gaussian-3](<https://github.com/jzy124/hw4/raw/master/pictures/1/test2-gaussian-3.png>)

![test2-gaussian-5](<https://github.com/jzy124/hw4/raw/master/pictures/1/test2-gaussian-5.png>)

![test2-gaussian-7](<https://github.com/jzy124/hw4/raw/master/pictures/1/test2-gaussian-7.png>)

###  2. 中值滤波器

中值滤波器是选取在规定大小的区域中按大小排序的中间值作为替代中心点位置像素的新像素值的平滑滤波器。中值滤波是图像处理中的一个常用步骤，它对于斑点噪声（speckle noise）和椒盐噪声（salt-and-pepper noise）来说尤其有用。

核心程序如下：

```python
for i in range(s, w + s):
    for j in range(s, h + s):
        for i1 in range(i - s, i + s + 1):
            for j1 in range(j - s, j + s + 1):
                arr_s[i1-(i - s)][j1-(j - s)] = arr[i1][j1]
        arr_f = arr_s.flatten()
        c = np.median(arr_f)
        img1[i-s][j-s] = c
```

下面是分别使用3×3、5×5和7×7大小的中值滤波器平滑test1和test2的结果。

![](<https://github.com/jzy124/hw4/raw/master/pictures/1/test1-median-3.png>)

![](<https://github.com/jzy124/hw4/raw/master/pictures/1/test1-median-5.png>)

![](<https://github.com/jzy124/hw4/raw/master/pictures/1/test1-median-7.png>)

![test2-median-3](<https://github.com/jzy124/hw4/raw/master/pictures/1/test2-median-3.png>)

![test2-median-5](<https://github.com/jzy124/hw4/raw/master/pictures/1/test2-median-5.png>)

![test2-median-7](<https://github.com/jzy124/hw4/raw/master/pictures/1/test2-median-7.png>)

###  3. 优缺点分析

高斯滤波器：

+ 优点：速度快，高斯滤波对于抑制服从正态分布的噪声效果非常好。

+ 缺点：虽然以模糊图像作为代价，可以看到test1图像中的白色条状噪声并没有被很好的去除。

中值滤波器：

+ 优点：对于斑点噪声处理效果非常好，test1中的白色条状噪声被很好的去除了。中值滤波器还有保存边缘的特性。
+ 缺点：图像模糊程度比高斯滤波器要更明显一些。

##  二. 空域高通滤波器

题目要求：利用高通滤波器滤波测试图像test3和4，包括unsharp masking，Sobel edge detector，Laplace edge detection，Canny algorithm。分析各自优缺点。

所有需要平滑图像的步骤使用的均是3×3的高斯滤波器。

###  1. unsharp masking

非锐化掩蔽的主要思想就是，用图像减去其平滑后的图像作为模板，然后把这个模板在加到原图像上做图像锐化（增强）。

核心代码：

```python
for i in range(w):
    for j in range(h):
        g_m[i][j] = img[i][j] - img1[i][j]
        img2[i][j] = img[i][j] + g_m[i][j]
        if img2[i][j]>255:
            img2[i][j]=255
        if img2[i][j]<0:
            img2[i][j]=0
```

下面是用非锐化掩蔽锐化之后的test3和test4图像：

![test3-unsharpMasking](<https://github.com/jzy124/hw4/raw/master/pictures/2/test3-unsharpMasking.png>)

![test4-unsharpMasking](<https://github.com/jzy124/hw4/raw/master/pictures/2/test4-unsharpMasking.png>)

###  2. Sobel edge detector

Sobel算子主要利用图像的一阶梯度来探测图像中物体的边缘，它有x和y两个方向的算子，分别是：       

G_x=[[-1,-2,-1],[0,0,0],[1,2,1]]

G_y=[[-1,0,1],[-2,0,2],[-1,0,1]]

用这两个矩阵分别与图像卷积就可以探测出x方向和y方向的边缘，再将二者合并就可以的到整幅图的边缘信息。

第一次测试的时候仍然是出现了雪花屏的错误结果，我分析的原因是，在卷积完成之后，由于Sobel算子中含有负数，所以像素点上也有可能出现负数，但是我没有对它取绝对值。

核心代码：

```python
for i in range(s, w + s):
    for j in range(s, h + s):
        for i1 in range(i - s, i + s + 1):
            for j1 in range(j - s, j + s + 1):
                arr_s[i1 - (i - s)][j1 - (j - s)] = arr[i1][j1]
        arr_f = arr_s.flatten()
        c_x = 0
        c_y = 0
        for k in range(pow(size, 2)):
            c_x = c_x + s_x[k] * arr_f[k]
            c_y = c_y + s_y[k] * arr_f[k]
        img_x[i - s][j - s] = abs(c_x)
        img_y[i - s][j - s] = abs(c_y)
```

下面是用Sobel算子对test3和test4图片进行边缘探测的结果：

![test3-Sobel](<https://github.com/jzy124/hw4/raw/master/pictures/2/test3-Sobel.png>)

![test4-Sobel](<https://github.com/jzy124/hw4/raw/master/pictures/2/test4-Sobel.png>)

### 3. Laplace edge detection

Laplace算子是利用图像的二阶差分来进行边缘探测的。考虑对角线的Laplace算子如下：

L=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]

用它与源图像进行卷积就可以的到图像的边缘信息。

下面是test3和test4进行Laplace edge detection的结果：

![test3-Laplace](<https://github.com/jzy124/hw4/raw/master/pictures/2/test3-Laplace.png>)

![test4-Laplace](<https://github.com/jzy124/hw4/raw/master/pictures/2/test4-Laplace.png>)

###  4. Canny algorithm

Canny算法首先也对图像计算了它的x方向和y方向的一阶梯度。然后给定了一个阈值范围，当梯度值大于阈值范围中大的那个数时，这点就被确定是边缘；若是小于较小的那个数，那么这个点就被确定不是边缘。当某点的梯度处于阈值范围内时，如果这个点连接一个大于最大阈值的像素点，那么就属于边界。

由于Canny算法比较复杂，在程序中我直接使用了OpenCV中自带的函数：

```python
img1 = cv2.Canny(img, 50, 150)
```

其中第二个参数和第三个参数就是设定的阈值。

下面是用Canny算法探测test3和test4图像的结果：

![test3-Canny](<https://github.com/jzy124/hw4/raw/master/pictures/2/test3-Canny.png>)

![test4-Canny](<https://github.com/jzy124/hw4/raw/master/pictures/2/test4-Canny.png>)

###  5. 优缺点分析

unsharp masking：

+ 优点：有效的对图像进行锐化。
+ 缺点：可能放大平坦区域的高频噪声。

Sobel edge detector：

+ 优点：计算简单，速度很快。
+ 缺点：算子对边缘定位不是很准确；只在两个方向上进行，对于有纹理的图片比较难处理。

Laplace edge detection：

+ 优点：一般增强技术对于陡峭的边缘和缓慢变化的边缘很难确定其边缘线的位置，但此算子却可用二次微分正峰和负峰之间的过零点来确定，对孤立点或端点更为敏感，因此特别适用于以突出图像中的孤立点、孤立线或线端点为目的的场合。
+ 缺点：拉普拉斯对噪声敏感，会产生双边效果，不能检测出边的方向，通常不直接用于边的检测。

Canny algorithm：

+ 对于边缘的探测精度很高，得到的是细化的边缘，噪声的抑制效果好。



##  参考资料

+ python+opencv均值滤波，高斯滤波，中值滤波，双边滤波：https://blog.csdn.net/qq_27261889/article/details/80822270
+ [Python图像处理] 四.图像平滑之均值滤波、方框滤波、高斯滤波及中值滤波：https://blog.csdn.net/Eastmount/article/details/82216380
+ Python 实现中值滤波、均值滤波：https://blog.csdn.net/Dooonald/article/details/78260299
+ 【OpenCV】边缘检测：Sobel、拉普拉斯算子：https://blog.csdn.net/xiaowei_cqu/article/details/7829481
+ OpenCV-Python教程（8、Canny边缘检测）：https://blog.csdn.net/sunny2038/article/details/9202641
+ Canny Edge Detection Tutorial（Canny 边缘检测教程）：http://www.360doc.com/content/13/0301/12/11644963_268624153.shtml





