import running_inference as ri
import os
import cv2
import numpy as np




dataset = 'test'
img = 'test/ct16x61778873235.5533102.jpg'
model = 'rf-detr_model_top-view.pth'
results_all = ri.detect_model(model,img)
output = 'output'
os.makedirs(output,exist_ok=True)


for result in results_all:
    filename = result["filename"]
    annotated_frame = result["annotated_frame"]
    masks = result["masks"]
    
    H,W = annotated_frame.shape[:2]
    zeros_mask = np.zeros((H,W,3),dtype=np.uint8)
    for i, mask in enumerate(masks):
        result_img = zeros_mask.copy()
        result_img[mask] = (255,0,0)
        ys,xs = np.where(mask)
        
        cv2.circle(result_img,(int(xs.mean()),int(ys.mean())),2,(0,255,0,2))
   

        cv2.imshow('teste',result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #output_path = os.path.join(output,filename)
    #cv2.imwrite(output_path,annotated_frame)

    
