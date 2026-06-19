import os
import cv2
import numpy as np
import running_inference as ri 
import pandas as pd 


model = 'rf-detr-SEGM-lateral.pth'
dataset = 'test'
calib_img = 'calibration_img.jpg'
output = 'output'
os.makedirs(output, exist_ok=True)

# imagem de calibração 

def calib(calib_path):
    
    results_all = ri.detect_model(model,calib_path)

    result = results_all[0]
    boxes = result["boxes"]

    if boxes is None or len(boxes) == 0:
        raise ValueError("Nenhuma bounding box detectada na calibração.")

    heights = boxes[:,3] - boxes[:,1]
    best_idx = int(np.argmax(heights))
    x1,y1,x2,y2 = boxes[best_idx]

    height_bb = y2 - y1 
    height_real = 23
    resolution = height_real/height_bb

    return resolution

#desgaste com detr

def ml_desgaste(images_path):
    resolução = calib(calib_img)
    alturas_dict = {}
    results_all = ri.detect_model(model,images_path)

    for result in results_all:

        filename = result["filename"]
        boxes = result["boxes"]
        annotated_frame = result["annotated_frame"]  

        if boxes is None or len(boxes) == 0:
            print(f"Sem detecção: {filename}")
            continue

        heights = boxes[:,3] - boxes[:,1]
        best_idx = int(np.argmax(heights))
        x1,y1,x2,y2 = boxes[best_idx]

        altura = (y2 - y1) * resolução
        alturas_dict[filename] = altura
        output_path = os.path.join(output,filename)

        H,W = annotated_frame.shape[:2]

        center = W // 2 + 77
        cv2.line(annotated_frame,(int(center),int(y2)),(int(center),int(y1)),(255,0,0),2)

        cv2.imwrite(output_path,annotated_frame)

    return alturas_dict

altura = ml_desgaste(dataset)
print(altura)

 