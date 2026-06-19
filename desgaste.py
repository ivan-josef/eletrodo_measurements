import os
import cv2
import numpy as np
import running_inference as ri 
import pandas as pd 
import calibracao_desgaste as cal


model = 'modelos/rf-detr-SEGM-lateral.pth'
calib_img = 'test_lateral-view/ct155_1.jpg'

def desgaste(images_path):
    resolução = cal.calib(calib_img) # ? 
    alturas_dict = []
    results_all = ri.detect_model(model,images_path,1296)

    for result in results_all:
        classe = result["classes"]
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
        alturas_dict.append({
            "classe":classe,
            "altura":altura
        })

        # H,W = annotated_frame.shape[:2]
        # center = W // 2 + 77
        # cv2.line(annotated_frame,(int(center),int(y2)),(int(center),int(y1)),(255,0,0),2)
        # cv2.imwrite(filename,annotated_frame)

    return alturas_dict


if __name__ == "__main__":
    dataset = 'test_lateral-view'
    desgaste(dataset)