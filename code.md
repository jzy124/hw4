###  源代码

####  1.

高斯滤波器

```python
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


size = 5
std = 1.5

s = (size - 1) // 2
g1 = creategaussian(size, std)
g = g1.flatten()

img = cv2.imread("test1.pgm", 0)
img1 = img.copy()
arr = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT, value=0)
w, h = img.shape
arr_s = np.zeros((size, size))
arr_f = np.zeros(pow(size, 2))

for i in range(s, w + s):
    for j in range(s, h + s):
        for i1 in range(i - s, i + s + 1):
            for j1 in range(j - s, j + s + 1):
                arr_s[i1-(i - s)][j1-(j - s)] = arr[i1][j1]
        arr_f = arr_s.flatten()
        c = 0
        for k in range(pow(size, 2)):
            c = c + g[k] * arr_f[k]
        img1[i-s][j-s] = c

plt.figure("test1-gaussian")
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("original:test1.pgm")
plt.axis('off')

plt.subplot(122)
plt.imshow(img1, cmap='gray')
plt.title("gaussian 5*5")
plt.axis('off')

plt.show()
```

中值滤波器

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

size = 7
s = (size - 1) // 2

img = cv2.imread("test1.pgm", 0)
img1 = img.copy()
arr = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT, value=0)
w, h = img.shape
arr_s = np.zeros((size, size))
arr_f = np.zeros(pow(size, 2))

for i in range(s, w + s):
    for j in range(s, h + s):
        for i1 in range(i - s, i + s + 1):
            for j1 in range(j - s, j + s + 1):
                arr_s[i1-(i - s)][j1-(j - s)] = arr[i1][j1]
        arr_f = arr_s.flatten()
        c = np.median(arr_f)
        img1[i-s][j-s] = c

plt.figure("test1-median")
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("original:test1.pgm")
plt.axis('off')

plt.subplot(122)
plt.imshow(img1, cmap='gray')
plt.title("gaussian 7*7")
plt.axis('off')

plt.show()
```

####  2.

Unsharp Masking

```python
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


size = 3
std = 1.5

s = (size - 1) // 2
g1 = creategaussian(size, std)
g = g1.flatten()

img = cv2.imread("test4.tif", 0)
img1 = img.copy()
img2 = img.copy()
g_m = img.copy()
arr = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT, value=0)
w, h = img.shape
arr_s = np.zeros((size, size))
arr_f = np.zeros(pow(size, 2))

for i in range(s, w + s):
    for j in range(s, h + s):
        for i1 in range(i - s, i + s + 1):
            for j1 in range(j - s, j + s + 1):
                arr_s[i1-(i - s)][j1-(j - s)] = arr[i1][j1]
        arr_f = arr_s.flatten()
        c = 0
        for k in range(pow(size, 2)):
            c = c + g[k] * arr_f[k]
        img1[i-s][j-s] = c

for i in range(w):
    for j in range(h):
        g_m[i][j] = img[i][j] - img1[i][j]
        img2[i][j] = img[i][j] + g_m[i][j]
        if img2[i][j]>255:
            img2[i][j]=255
        if img2[i][j]<0:
            img2[i][j]=0

plt.figure("test4-unsharpMasking")
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("original:test4.tif")
plt.axis('off')

plt.subplot(122)
plt.imshow(img2, cmap='gray')
plt.title("unsharpMasking")
plt.axis('off')

plt.show()
```

Sobel Edge Detector

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

size = 3
s = (size - 1) // 2

s_x = [-1, -2, -1,
       0, 0, 0,
       1, 2, 1]
s_y = [-1, 0, 1,
       -2, 0, 2,
       -1, 0, 1]

img_o = cv2.imread("test4.tif", 0)
img_x = img_o.copy()
img_y = img_o.copy()
img = cv2.GaussianBlur(img_o, (3, 3), 0)
arr = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT, value=0)
w, h = img.shape
arr_s = np.zeros((size, size))
arr_f = np.zeros(pow(size, 2))

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

max_x = img_x.max()
max_y = img_y.max()

for i in range(w):
    for j in range(h):
        img_x[i][j] = int(img_x[i][j] / max_x * 255)
        img_y[i][j] = int(img_y[i][j] / max_y * 255)

img1 = cv2.addWeighted(img_x, 0.5, img_y, 0.5, 0)

plt.figure("test4-Sobel")
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title("original:test4.tif")
plt.axis('off')

plt.subplot(222)
plt.imshow(img_x, cmap='gray')
plt.title("Grad-X")
plt.axis('off')

plt.subplot(223)
plt.imshow(img_y, cmap='gray')
plt.title("Grad-Y")
plt.axis('off')

plt.subplot(224)
plt.imshow(img1, cmap='gray')
plt.title("Sobel edge detector")
plt.axis('off')

plt.show()
```

 Laplace Edge Detection

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

size = 3
s = (size - 1) // 2

la = [-1, -1, -1,
      -1, 8, -1,
      -1, -1, -1]

img_o = cv2.imread("test3_corrupt.pgm", 0)
img1 = img_o.copy()
img = cv2.GaussianBlur(img_o, (3, 3), 0)
arr = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT, value=0)
w, h = img.shape
arr_s = np.zeros((size, size))
arr_f = np.zeros(pow(size, 2))

for i in range(s, w + s):
    for j in range(s, h + s):
        for i1 in range(i - s, i + s + 1):
            for j1 in range(j - s, j + s + 1):
                arr_s[i1 - (i - s)][j1 - (j - s)] = arr[i1][j1]
        arr_f = arr_s.flatten()
        c = 0
        for k in range(pow(size, 2)):
            c = c + la[k] * arr_f[k]
        img1[i - s][j - s] = abs(c)

maxi = img1.max()

for i in range(w):
    for j in range(h):
        img1[i][j] = int(img1[i][j] / maxi * 255)

plt.figure("test3-Laplace")
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("original:test3_corrupt.pgm")
plt.axis('off')

plt.subplot(122)
plt.imshow(img1, cmap='gray')
plt.title("Laplace edge detection")
plt.axis('off')

plt.show()
```

Canny Algorithm

```python
import cv2
import matplotlib.pyplot as plt

img_o = cv2.imread("test4.tif", 0)

img = cv2.GaussianBlur(img_o, (3, 3), 0)
img1 = cv2.Canny(img, 50, 150)

plt.figure("test4-Canny")
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("original:test4.tif")
plt.axis('off')

plt.subplot(122)
plt.imshow(img1, cmap='gray')
plt.title("Canny algorithm")
plt.axis('off')

plt.show()
```