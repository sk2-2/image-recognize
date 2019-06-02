import cv2
import numpy as np
from sklearn import svm
img2=cv2.imread("c1.png")
img3=cv2.imread("c2.png")
img4=cv2.imread("x1.png")
img5=cv2.imread("x2.png")
img6=cv2.imread("c3.png")

def process(img1):
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("c1.png",img1)
    img1=img1/255
    #print(img1)
    img=np.identity(3)
    a=img1.shape
    l=[]
    global l1
    l1=[]
    m=0

    for i in range(0,18,3):
        for j in range(0,18,3):
            a=np.dot(img,img1[i:i+3,j:j+3])
            b=a.mean()
            l.append(b)
    #print(l)
    #print(a,l)
    c1=(np.asarray(l))
    c1=c1.reshape(6,6)
    #print(c1,c1.shape)

    for i in range(0,3,1):
        for j in range(0,3,1):
            a1=np.dot(img,c1[i:i+3,j:j+3])
            b1=a.mean()
            l1.append(b1)
    #print(l1)
    return l1
process(img2)
a1=l1
process(img3)
a2=l1
process(img4)
a3=l1
process(img5)
a4=l1
process(img6)
a5=l1


print(a1,"--------------",a2,"--------------",a3,"--------------",a4)

clf=svm.SVC(kernel="linear",C=3)
f=[a1,a2,a3,a4]
l=["O","O","X","X"]
t=clf.fit(f,l)
r=t.predict([a5])
print(r)

        
        
                
             
