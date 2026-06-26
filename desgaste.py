
import numpy as np
import running_inference as ri 
import dicionario_ref as ref
import cv2
from model_manager import manager

def desgaste(frame,resolution,annotate=False,crop_box = None):
    resolucao = resolution
    alturas_dict = []

    img = cv2.imread(frame)

    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {frame}") 

    if crop_box is not None:
        x1,y1,x2,y2 = crop_box
        img = img[y1:y2,x1:x2]


    manager.use('modelos/rf-detr-SEGM-lateral.pth')

    results_all = ri.detect_model(
        img, annotate=annotate
    )
    
    result = results_all[0]
    classe = result["classes"]
    boxes = result["boxes"]

    if boxes is None or len(boxes) == 0:
        return print(f"Sem detecção")
        

    heights = boxes[:,3] - boxes[:,1]
    best_idx = int(np.argmax(heights))
    x1,y1,x2,y2 = boxes[best_idx]

    altura = (y2 - y1) * resolucao
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
    desgaste()