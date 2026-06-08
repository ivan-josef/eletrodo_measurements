import os
import cv2
import numpy as np
from rfdetr import RFDETRSegMedium
import running_inference as ri 
import white_pixels as white

model = 'rf-detr-SEGM-lateral.pth'
dataset = 'test'
calib_img = 'calibration_img.jpg'
output = 'output'
os.makedirs(output, exist_ok=True)

# imagem de calibração 

def calib(calib_path):
    calib_img = cv2.imread(calib_path) 
    boxes,masks,scores,classes,frame_anotated = ri.detect_model(model,calib_path)

    for box in boxes:
        x1,y1,x2,y2 = box 

    height_bb = y2 - y1 
    height_real = 23
    resolution = height_real/height_bb
    return resolution

# desgaste com detr

def ml_desgaste(images_path):
    resolução = calib(calib_img)
    alturas_dict = {}
    for f in os.listdir(images_path):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):        
            img_path = os.path.join(dataset,f)
            boxes,masks,scores,classes,frame_anotated = ri.detect_model(model,img_path)
            for box in boxes:
                x1,y1,x2,y2 = box 
            altura = (y2 - y1) * resolução
            alturas_dict[f] = altura
            output_path = os.path.join(output,f)

            H,W = frame_anotated.shape[:2]

            center = W // 2 + 77
            print(center)
            cv2.line(frame_anotated,(int(center),int(y2)),(int(center),int(y1)),(255,0,0),2)

            cv2.imwrite(output_path,frame_anotated)
    return alturas_dict

# altura = ml_desgaste(dataset)
#(np.float32(221.0), np.float32(160.0)

#desgaste com opencv

img = cv2.imread('test/ct155_5.jpg')
H,W = img.shape[:2]
center = W//2 + 80

lower = white.lower
up = white.up
img_csv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

white_mask = cv2.inRange(img_csv,lower,up)
cv2.imshow('teste',white_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# x = H
# for p in range(H):
#     cv2.circle(img,(center,x),2,(255,0,0),2)
#     x-=1
        




