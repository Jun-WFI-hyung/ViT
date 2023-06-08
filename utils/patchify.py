import cv2, torch
import numpy as np
from matplotlib import pyplot as plt


# N * N 그리드로 패치를 분할 할 때
# N의 배수 size인 정사각형 이미지를 매개변수로 받아야한다.
def patchify(img:np.array, N:int) -> torch.Tensor:
    x = torch.Tensor(np.array(img, dtype=np.float32))

    # stride = 1개 패치의 사이즈
    stride = x.shape[0] // N

    x = x.unfold(dimension=0, size=stride, step=stride)
    x = x.unfold(dimension=1, size=stride, step=stride)

    x = x.reshape(-1,3,stride,stride)
    x = x.permute(0,2,3,1)

    return x


def flatten(img:torch.Tensor) -> torch.Tensor:
    return img.reshape(img.shape[0], -1)

    

if __name__ == "__main__":
    is_image = 0
    devide = 3
    
    if is_image:
        img = cv2.imread("./download.jpeg")
        print(img.shape)
        test = patchify(img=img, N=devide).numpy().astype(np.uint8)

        for i in range(test.shape[0]):
            plt.subplot(3,3,i+1)
            plt.imshow(cv2.cvtColor(test[i, ...], cv2.COLOR_BGR2RGB))
        plt.show()
        
    else:
        test_mat = np.array([[[i]*3 for i in range(1+10*j, 10+10*j)] for j in range(1, 10)], dtype=np.uint8)        
        test = patchify(img=test_mat, N=devide)
        print(test.shape)
        
        test = flatten(test)
        print(test)
    