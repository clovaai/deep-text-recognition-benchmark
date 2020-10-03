import cv2
import numpy as np
import os
import imutils

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.png', gray)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, thresh = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(
    #     gray, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('./thresheld.png', thresh)
    return thresh

def threshold_img(input_img):
    _, thresh = cv2.threshold(input_img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_horizontal_lines(binarised_img):
    # Specify size on horizontal axis
    horizontal = np.copy(binarised_img)
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    
    return horizontal

# def remove_redacted_region(binarised_img):

    
def find_contours(img):
    h, w = img.shape[:2]
    # mask = np.zeros((h, w), np.uint8)
    _, contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # print(contours)
    print('Number of contours : {}'.format(len(contours)))
    # cnt = max(contours, key=cv2.contourArea)
    # img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.drawContours(img, [cnt], 0, 255, -1)
    return contours
    # return img

def find_signal_lines(input_img_dir):
    original_img = cv2.imread(input_img_dir)
    binarised_img = preprocess_image(input_img_dir)
    horizontal = np.copy(binarised_img)
    cols = horizontal.shape[1]
    horizontal_size = cols // 50
    print('Horizontal size is {}'.format(horizontal_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    dilated = cv2.dilate(binarised_img, dilate_kernel, iterations=1)
    cv2.imwrite('Dilated.png', dilated)
    contours = find_contours(dilated)
    filtered_contours = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if w / cols > 0.5:
            filtered_contours.append([x, y, w, h])
            # print('!!!')
    filtered_contours.sort(key=lambda c : c[2], reverse=True)
    filtered_contours = filtered_contours[2:5]
    for contour in filtered_contours:
        print(contour)
        [x, y, w, h] = contour
        # Shrink contour
        binarised_crop = binarised_img[y : y + h, x : x + w]
        shrinked_left = 0
        shrinked_right = binarised_crop.shape[1] - 1
        for col in range(binarised_crop.shape[1]):
            if np.count_nonzero(binarised_crop[:, col]) < 10: 
                shrinked_left += 1
            else:
                break
        for col in range(binarised_crop.shape[1] - 1, -1, -1):
            if np.count_nonzero(binarised_crop[:, col]) < 10:
                shrinked_right -= 1
            else:
                break
        binarised_crop = binarised_crop[:, shrinked_left : shrinked_right]
        cv2.imshow("Crop", binarised_crop)
        cv2.waitKey(0)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return original_img
    

if __name__ == "__main__":
    # preprocess_image('./input/CHS197_Redacted.png')
    input_img_dir = os.path.expanduser('~/datasets/ecg/png_final_August/CHS194_Redacted.png')
    dilated = find_signal_lines(input_img_dir)
    cv2.imwrite("contours.png", dilated)

    # horizontal = extract_horizontal_lines(thresheld_img)
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 3))
    # horizontal = cv2.morphologyEx(
    #     horizontal, cv2.MORPH_DILATE, dilate_kernel, iterations=5)
    # cv2.imwrite('horizontal.png', horizontal)
    # hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    # cv2.imwrite('HSV.png', hsv[:,0,:])
    # thresheld_hsv = threshold_img(hsv[:,0,:])
    # cv2.imwrite('threshold_hsv.png', thresheld_hsv)

    # blurred = cv2.GaussianBlur(original_img, (5, 5), 0)
    # cv2.imwrite('blurred.png', blurred)
    # thresheld_img = preprocess_image('resized.png')
    # contours = find_contours(dilated)
    # for contour in contours:
    #     [x, y, w, h] = cv2.boundingRect(contour)
    #     cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # # img_with_contours = cv2.drawContours(original_img, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite('contours.png', original_img)