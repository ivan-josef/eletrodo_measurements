import os
import cv2
import numpy as np
from rfdetr import RFDETRSegMedium
import running_inference as ri 
import pandas as pd 
import white_pixels as branco 


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

#desgaste com detr

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
            cv2.line(frame_anotated,(int(center),int(y2)),(int(center),int(y1)),(255,0,0),2)

            cv2.imwrite(output_path,frame_anotated)
    return alturas_dict

altura = ml_desgaste(dataset)
print(altura)

# #desgaste com opencv
# lut_csv = pd.read_csv('white_pixels_uniq.csv')
# output_2 = 'output_2'
# for f in os.listdir(dataset):
#     if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
#         continue
#     file = os.path.join(dataset,f)
#     img = cv2.imread(file)
#     img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#     lower = branco.lower
#     up = branco.up

#     white_mask = cv2.inRange(img_hsv, lower, up)
#     white_mask = cv2.medianBlur(white_mask,19)

#     img_mask = cv2.bitwise_and(img,img,mask=white_mask)


#     H,W = img_mask.shape[:2]
#     center = W//2 + 80
#     ys = np.where(np.any(white_mask != 0, axis=1))[0]
#     if len(ys) == 0:
#         print(f'Sem detecção: {f}')
#         continue

#     altura_real_ref = 22
#     altura_medida_ref = 1247 # ys[-1] - ys[0]
#     resolution = altura_real_ref / altura_medida_ref

#     altura_medida = resolution * (ys[-1] - ys[0]) 

#     for y in ys:
#         cv2.circle(img_mask, (center, y), 1, (255,0,0), 1)

#     print(f'altura é {f} é {altura_medida} e ys[-1] - ys[0] é {ys[-1] - ys[0]}')


#     output_path_2 = os.path.join(output_2,f)
#     cv2.imwrite(output_path_2,img_mask)
   




 