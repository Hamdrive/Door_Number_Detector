
from cv2 import cv2
import numpy as np
import easyocr

img_source = "door_3.jpg"
img = cv2.imread(img_source)
img_copy = img
cv2.imshow("Original_Image", img)

# Prepare image for OCR
gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
bfilter = cv2.bilateralFilter(gray, 10, 15, 15)
edge_image = cv2.Canny(bfilter, 30, 200)


# EasyOCR to extract the number
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(gray)
door_number = result[0][1]
print(door_number)
location = result[0][0]
print(location)


# To convert list into numpy array
contours = np.array(location).reshape((-1,1,2)).astype(np.int32)
print(contours)

# Crop out the relevant section
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [contours], 0,255,-1)
new_image = cv2.bitwise_and(img_copy, img_copy, mask=mask)

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

# Render result
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img_copy, text=door_number, org=(contours[0][0][0], contours[1][0][1]+100), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img_copy, tuple(contours[0][0]), tuple(contours[2][0]), (0,255,0),3)

# Display result
cv2.imshow("Door_Number_Detection_Image", res)
cv2.waitKey(0)

