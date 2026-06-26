import running_inference as ri
import numpy as np
import cv2


def calib(calib_path):

    img = cv2.imread(calib_path)
    
    results_all = ri.detect_model(img)

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

if __name__ == "__main__":
    calib()