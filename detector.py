import numpy as np
import cv2
import scipy.ndimage as ndimage
from math import log10, sqrt
import matplotlib.pyplot as plt

def read_img(img_name:str) -> np.ndarray:
    """
    Function to read an image and convert the image to black and white

    Parameters
    ----------
    img_name : str
        Name of the image to be read

    Returns
    -------
    np.ndarray
        Image array

    """
    image = cv2.imread(img_name, cv2.IMREAD_COLOR)
    return image


def theta(image: np.ndarray) -> np.ndarray:
    """
    Function to calculate the theta value array of the input image, which is further used to calculate the angle at which non maximum suppression occurs

    Parameters
    ----------
    image : np.ndarray
        Sobel operated image array

    Returns
    -------
    theta : np.ndarray
        Calculated theta value array of the input image to be used for non maximum suppression

    """
    duplicate_image = image.copy()
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float64)
    Ix = ndimage.filters.convolve(duplicate_image, Kx)
    Iy = ndimage.filters.convolve(duplicate_image, Ky)
    theta = np.arctan2(Iy, Ix)
    return theta


def non_max_suppression(img: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Function to perform non maximum suppression

    Parameters
    ----------
    img : np.ndarray
        Sobel operated image array
    D : np.ndarray
        Calculated theta value array

    Returns
    -------
    np.ndarray
        Non maximum suppression image array

    """
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError:
                pass

    return Z


def threshold(img: np.ndarray) -> np.ndarray:
    """
    Function to perform threshold operation on the input image
    

    Parameters
    ----------
    img : np.ndarray
        Non maximum suppression image array

    Returns
    -------
    np.ndarray
        Thresholded image array

    """
    weak_pixel = 75
    strong_pixel = 255
    lowthreshold = 0.05
    highthreshold = 0.15

    highThreshold = img.max() * highthreshold
    lowThreshold = highThreshold * lowthreshold

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def hysteresis(img: np.ndarray) -> np.ndarray:
    """
    Fuction to perform hystersis operation on the input image

    Parameters
    ----------
    img : np.ndarray
        Thresholded image array

    Returns
    -------
    np.ndarray
        Hysterisis image array

    """
    weak_pixel = 75
    strong_pixel = 255

    M, N = img.shape
    weak = weak_pixel
    strong = strong_pixel

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if (
                        (img[i + 1, j - 1] == strong)
                        or (img[i + 1, j] == strong)
                        or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong)
                        or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong)
                        or (img[i - 1, j] == strong)
                        or (img[i - 1, j + 1] == strong)
                    ):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError:
                    pass

    return img


def psnr(img1: np.ndarray, img2: np.ndarray) -> int:
    """
    Rudimentary function to calculate the PSNR value

    Parameters
    ----------
    img1 : np.ndarray
        Original image
    img2 : np.ndarray
        Processed image

    Returns
    -------
    int
        PSNR value

    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / sqrt(mse))


def sobel_with_blur(img: np.ndarray) -> np.ndarray:
    """
    Function that blurs the image for furthur processing and also applies sobel operation to the image
    

    Parameters
    ----------
    img : np.ndarray
        Original image array

    Returns
    -------
    grad : TYPE
        Sobel operated image array

    """
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    PSNR_try = cv2.blur(img, (5, 5))
    PSNR_try2 = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    PSNR_try2 = cv2.blur(PSNR_try2, (5, 5))

    PSNR_value1 = psnr(img, PSNR_try)
    PSNR_value2 = psnr(img, PSNR_try2)

    if PSNR_value1 > PSNR_value2:
        src = cv2.blur(img, (5, 5))

    elif PSNR_value1 < PSNR_value2:
        src = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        src = cv2.blur(src, (5, 5))

    else:
        src = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        src = cv2.blur(src, (5, 5))

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(
        gray,
        ddepth,
        1,
        0,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv2.BORDER_DEFAULT,
    )
    grad_y = cv2.Sobel(
        gray,
        ddepth,
        0,
        1,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv2.BORDER_DEFAULT,
    )

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


def bright(orig_img: np.ndarray, hys_img: np.ndarray) -> np.ndarray:
    """
    Function that makes the edges more prominent to the system

    Parameters
    ----------
    orig_img : np.ndarray
        Original image array
    hys_img : np.ndarray
        Hysterisis image array

    Returns
    -------
    np.ndarray
        Final edge detected image array

    """
    brightness_values = []
    x = 0.5

    for num in range(60):
        brightness_values.append(x)
        x += 0.5

    hys_img = cv2.cvtColor(hys_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for values in enumerate(brightness_values):
        mod_img = cv2.add(hys_img, np.array([values[1]]))

        try:
            mod_img = cv2.cvtColor(mod_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        except:
            pass

        psnr_orig = psnr(orig_img, hys_img)
        psnr_mod = psnr(orig_img, mod_img)

        if psnr_orig > psnr_mod:
            hys_img = hys_img

        elif psnr_orig < psnr_mod:
            hys_img = mod_img

        else:
            hys_img = mod_img

    return hys_img

def Luminance_ed(img_name: str) -> np.ndarray:
    """
    Main method for Luminance Edge Detector

    Parameters
    ----------
    img_name : str
        Original image file name

    Returns
    -------
    np.ndarray
        Final edge detected image array

    """
    image = read_img(img_name)
    sobel_img = sobel_with_blur(image)
    theta_value = theta(sobel_img)
    non_max_suppress_img = non_max_suppression(sobel_img, theta_value)
    threshold_img = threshold(non_max_suppress_img)
    hys_img = hysteresis(threshold_img)
    final_img = bright(image, hys_img)
    return final_img

def trial():
    """
    Trial method to demonstrate edge detection using the included orig.jpg mri brain tumour image

    Returns
    -------
    None

    """
    ed_img = Luminance_ed("orig.jpg")
    plt.imshow(ed_img, 'gray')
    plt.show()