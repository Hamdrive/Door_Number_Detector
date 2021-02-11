# Door_Number_Detector
Detects the number present in the image given, and returns the number using Python and Easy OCR (https://jaided.ai/easyocr/)

Weights for detecting doors and handles with YOLO can be downloaded from: [YOLO_weights](https://drive.google.com/open?id=1i9E9pTPN5MtRxgBJWLnfQl2ypCv92dXk) (mAP=45%). For running YOLO you might also need the network configuration file [yolo-obj.cfg](https://github.com/MiguelARD/DoorDetect-Dataset/blob/master/yolo-obj.cfg) and a text file where the detected classes names and their order is specified [obj.names](https://github.com/MiguelARD/DoorDetect-Dataset/blob/master/obj.names).

Git Repos referred to: 
https://github.com/MiguelARD/DoorDetect-Dataset (For door dataset)
https://github.com/misbah4064/yolo_objectDetection_imagesCPU (For YOLO image detection)
