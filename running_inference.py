import cv2
import supervision as sv
import os 
from rfdetr import RFDETRSegMedium
from PIL import Image
import numpy as np 


def separate_img(path):
    images = []
    if os.path.isdir(path):
        for f in os.listdir(path):
            file = os.path.join(path,f)  
            img = cv2.imread(file)
            images.append(img)
        return images
    else:
        img = cv2.imread(path)
        images.append(img)
        return images
            
            
def detect_model(model, path):
    
    data = separate_img(path)
    H,W = data[0].shape[:2]
    


    modelo = RFDETRSegMedium(pretrain_weights=model)
    modelo.optimize_for_inference()


    for img in data:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img).resize((640,640))
        results = modelo.predict(pil_img,threshold=0.5)

        scale_y = H / 640
        scale_x = W / 640
        
        boxes = results.xyxy.copy()

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        results.xyxy = boxes


        if results.mask is not None:
            masks_resized = []
            for mask in results.mask:
                mask = mask.astype("uint8") * 255

                mask = cv2.resize(
                    mask,
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                )

                masks_resized.append(mask.astype(bool))
                results.mask = np.array(masks_resized)

        scores = results.confidence.tolist()
        classes = results.class_id.tolist()

        # inference 

        labels = [
            f'{class_id} {conf}'
            for class_id, conf in zip(results.class_id,results.confidence)
                  ]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        annotated_frame = sv.MaskAnnotator().annotate(img,results)
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame,results)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame,results,labels)


        return boxes, masks_resized, scores, classes, annotated_frame