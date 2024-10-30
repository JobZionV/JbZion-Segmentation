import detector as ed

import cv2
import cv2 as cv2
import numpy as np
from scipy.signal import find_peaks
         
def threshold_mod(res):    
    image = res.copy()
    count, division = np.histogram(image, bins=256)
    peaks, props = find_peaks(count, prominence=(100,1000))
    peaks = peaks.tolist()
    prominences = props["prominences"].tolist()
    dict1 = {}
    dict2 = {}
    for x in range(len(prominences)):
        if prominences[x]>500:
            dict1[prominences[x]] = x
        else:
            dict2[prominences[x]] = x
    list1 = list(dict1)
    list2 = list(dict2)
    list3 = []
    list4 = []

    for x in list1:
        x = int(x)
        list3.append(x)

    for x in list2:
        x = int(x)
        list4.append(x)

    list1 = list3
    list2 = list4

    list1.sort(reverse=True)
    list2.sort(reverse=True)

    a = True
    b = True
    c = True
    d = True

    for x in list1:
        num = dict1[x]
        peak = peaks[num]
        if 120<peak<200:
            _, threshold = cv2.threshold(image, peak, 255, cv2.THRESH_BINARY)
            a = False
            b = False
            c = False
            d = False
            break

    if len(list2) != 0:
        if a:
            num1 = max(list2)
            num2 = dict2[num1]
            peak = peaks[num2]
            if 120<peak<200:
                _, threshold = cv2.threshold(image, peak, 255, cv2.THRESH_BINARY)
                b = False
                c = False
                d = False

    if b:
        for x in range(len(list2)-1):
            num1 = list2[x+1]
            num2 = dict2[num1]
            peak = peaks[num2]
            if 130<peak<200:
                _, threshold = cv2.threshold(image, peak, 255, cv2.THRESH_BINARY)
                c = False
                d = False
                break

    if d:
        for x in range(len(list2)-1):
            num1 = list2[x+1]
            num2 = dict2[num1]
            peak = peaks[num2]
            if 100<peak<200:
                _, threshold = cv2.threshold(image, peak, 255, cv2.THRESH_BINARY)
                c = False
                break

    if c:
        _, threshold = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    try:
        threshold = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
        
    except:
        pass
    
    try:
        threshold = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
        
    except:
        pass
    
    return threshold

def JbZion(img_name):
    img = cv2.imread(img_name)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    morph = cv2.morphologyEx(img, cv2.MORPH_OPEN,kernel)
    _,t = cv2.threshold(morph, 50, 255, cv2.THRESH_BINARY_INV)
    
    t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)

    threshold = threshold_mod(img)

    edges = ed.detect(img_name)[0]

    edges_gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    dst = cv2.addWeighted(threshold, 0.5, edges_gray, 0.5, 0.0)
    
    dst = cv2.subtract(dst,t)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    res = cv2.morphologyEx(dst,cv2.MORPH_OPEN,kernel)

    th, res = cv2.threshold(res, 125, 255, cv2.THRESH_BINARY)

    img1 = res.copy()
    
    orig_img = cv2.imread(img_name)

    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)

    colormask = np.zeros(orig_img.shape, dtype=np.uint8)
    colormask[thresh!=0] = np.array((0,0,255))

    ret, markers = cv2.connectedComponents(thresh)

    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
    if len(marker_area) == 0:
        marker_area.append(1)
    largest_component = np.argmax(marker_area)+1
    brain_mask = markers==largest_component

    brain_out = orig_img.copy()
    brain_out[brain_mask==False] = (0,0,0)

    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((8,8),np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    brain_out = orig_img.copy()
    brain_out[closing==False] = (0,0,0)

    brain_out = cv2.threshold(brain_out, 10,255, cv2.THRESH_BINARY_INV)[1]

    final_img = cv2.subtract(img1, cv2.cvtColor(brain_out, cv2.COLOR_BGR2GRAY))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    final_img = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
    final_img = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    final_img = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    final_img = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    final_img = cv2.morphologyEx(final_img, cv2.MORPH_OPEN, kernel)  
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    final_img = cv2.morphologyEx(final_img, cv2.MORPH_CLOSE, kernel)
    return final_img