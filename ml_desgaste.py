import os
import cv2
import numpy as npq
from rfdetr import RFDETRSegMedium
import running_inference as ri 

model = 'rf-detr-SEGM-lateral.pth'
output = 'output'
os.makedirs(output, exist_ok=True)

# imagem de calibração 

def calib(calib_path):
    calib_img = cv2.imread(calib_path) 
    H,W = calib_img.shape[:2]
    boxes,masks,scores,classes,frame_anotated = ri.detect_model(model,calib_path)

    for box in boxes:
        x1,y1,x2,y2 = box 

    height_bb = y2 - y1 
    height_real = 23
    resolution = height_real/height_bb
    return resolution



dataset = 'test'
resolução = calib('calibration_img.jpg')
alturas_dict = {}
for f in os.listdir(dataset):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):        
        img_path = os.path.join(dataset,f)
        boxes,masks,scores,classes,frame_anotated = ri.detect_model(model,img_path)
        for box in boxes:
            x1,y1,x2,y2 = box 
        altura = (y2 - y1) * resolução
        alturas_dict[f] = altura
        output_path = os.path.join(output,f)
        cv2.imwrite(output_path,frame_anotated)

print(alturas_dict)


