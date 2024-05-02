import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt


def LPF_processing(img):
    """
    將原圖過濾高頻部分
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[0:int(h/5), 0:int(w/5)] = 1  # 建立遮罩，長寬是原本影像的 1 / 5，位置是以左上角為起點
    img = img*mask  # 遮罩的地方乘1，也就讓低頻的部分通過(高頻部分都會因為乘上0而被捨棄掉)

    spectrum = 20*np.log(np.abs(img))  # 將經過LPF之後的影像轉為頻譜
    return img, spectrum


if __name__ == "__main__":
    # 讀入影像
    img = cv.imread('./input/turing.jpg', 0)

    # DCT
    # 先將原圖轉換成 float32 格式才能使用 opencv 的 dct function
    dct_img = cv.dct(np.float32(img))
    dct_magnitude_spectrum = 20*np.log(np.abs(dct_img))  # 透過同樣手法轉換為頻譜

    # LPF 實作
    LPF_img, LPF_dct_magnitude_spectrum = LPF_processing(dct_img)

    # 將經過LPF之後的頻譜轉回圖像
    idct_img = cv.idct(LPF_img)

    # FT
    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)
    ft_magnitude_spectrum = 20*np.log(np.abs(ft_shift))

    # 將 array 畫成圖像
    plt.figure(figsize=(5, 6))  # 設定輸出影像長寬

    plt.subplot(221), plt.imshow(ft_magnitude_spectrum, cmap='gray')  # FT 頻譜圖
    plt.title('Spectrum after FT'), plt.xticks([]), plt.yticks([])

    plt.subplot(222), plt.imshow(
        dct_magnitude_spectrum, cmap='gray')  # DCT 頻譜圖
    plt.title('Spectrum after DCT'), plt.xticks([]), plt.yticks([])

    plt.subplot(223), plt.imshow(
        LPF_dct_magnitude_spectrum, cmap='gray')  # 透過遮罩做 LPF 的頻譜圖
    plt.title('DCT after LPF'), plt.xticks([]), plt.yticks([])

    plt.subplot(224), plt.imshow(idct_img.astype(np.uint8), cmap='gray')
    plt.title('Image after LPF (IDCT)'), plt.xticks([]), plt.yticks([])

    # 儲存圖像
    plt.savefig('./output/Result.png', bbox_inches='tight', dpi=300)
