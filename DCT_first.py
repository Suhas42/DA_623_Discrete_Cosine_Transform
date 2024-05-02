import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt

def LPF_processing(img):
    """
    Filtering out high-frequency components of the original image
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[0:int(h/5), 0:int(w/5)] = 1  # Create a mask with dimensions 1/5 of the original image, positioned at the top left corner
    img = img*mask  # Multiply areas covered by the mask by 1, allowing low-frequency components to pass (high-frequency components are discarded because they are multiplied by 0)

    spectrum = 20*np.log(np.abs(img))  # Convert the image after LPF to a spectrum
    return img, spectrum

if __name__ == "__main__":
    # Read the image
    img = cv.imread('./input/turing.jpg', 0)

    # DCT
    # Convert the original image to float32 format to use the dct function in OpenCV
    dct_img = cv.dct(np.float32(img))
    dct_magnitude_spectrum = 20*np.log(np.abs(dct_img))  # Convert it to spectrum using the same method

    # LPF Implementation
    LPF_img, LPF_dct_magnitude_spectrum = LPF_processing(dct_img)

    # Convert the spectrum after LPF back to image
    idct_img = cv.idct(LPF_img)

    # FT
    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)
    ft_magnitude_spectrum = 20*np.log(np.abs(ft_shift))

    # Plot the arrays as images
    plt.figure(figsize=(5, 6))  # Set output image dimensions

    plt.subplot(221), plt.imshow(ft_magnitude_spectrum, cmap='gray')  # FT spectrum plot
    plt.title('Spectrum after FT'), plt.xticks([]), plt.yticks([])

    plt.subplot(222), plt.imshow(
        dct_magnitude_spectrum, cmap='gray')  # DCT spectrum plot
    plt.title('Spectrum after DCT'), plt.xticks([]), plt.yticks([])

    plt.subplot(223), plt.imshow(
        LPF_dct_magnitude_spectrum, cmap='gray')  # DCT spectrum plot after LPF using mask
    plt.title('DCT after LPF'), plt.xticks([]), plt.yticks([])

    plt.subplot(224), plt.imshow(idct_img.astype(np.uint8), cmap='gray')
    plt.title('Image after LPF (IDCT)'), plt.xticks([]), plt.yticks([])

    # Save the image
    plt.savefig('./output/Result.png', bbox_inches='tight', dpi=300)
