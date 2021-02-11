from cv2 import cv2
import numpy as np
import easyocr

# Required files
yolo_weight = "YOLO_wts_&_confgs/yolo-obj.weights"
yolo_confg = "YOLO_wts_&_confgs/yolo-obj.cfg"
yolo = cv2.dnn.readNet(yolo_weight, yolo_confg)
classes = []

# Obtain class names
obj_names = "YOLO_wts_&_confgs/obj.names"
with open(obj_names, "rt") as file:
    classes = [line.strip() for line in file.readlines()]
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

# Loading Images
name = "Images/door_10.jpg"
img = cv2.imread(name)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo.setInput(blob)
outputs = yolo.forward(output_layers)

# Checking if detection meets criteria and storing coordinates
class_ids = []
confidences = []
boxes = []
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1: #Ideally should work with higher tolerance
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Re-init for door/apt number detection
img_copy = img
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

# Render result for door/apt number detection
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img_copy, text=door_number, org=(contours[0][0][0], contours[1][0][1]+150), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img_copy, tuple(contours[0][0]), tuple(contours[2][0]), (0,255,0),3)

# Render result for door & handle detection
colorRed = (0,0,255)
colorGreen = (0,255,0)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
        cv2.putText(img, label, (x, y + 100), cv2.FONT_HERSHEY_PLAIN, 2, colorRed, 3)

# Display/Save result
#cv2.imshow("Image", img)
cv2.imwrite("output.jpg",res)
#cv2.waitKey(0)
#cv2.destroyAllWindows()