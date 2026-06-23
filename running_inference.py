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
            if img is not None:
                images.append((f, img))
    else:
        img = cv2.imread(path)
        if img is not None:
            images.append((os.path.basename(path),img))

    return images
                        

def detect_model(model, path):
    
    data = separate_img(path)
    
    modelo = RFDETRSegMedium(pretrain_weights=model,resolution=1296)
    modelo.optimize_for_inference()

    results_all = []

    for filename,img in data:
        H,W = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((640,640))
        results = modelo.predict(pil_img,threshold=0.5)

        scale_y = H / 640
        scale_x = W / 640

        if results.xyxy is not None:        
            boxes = results.xyxy.copy()

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            results.xyxy = boxes

        
        masks_resized = []
        if results.mask is not None:
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
        annotated_frame = sv.MaskAnnotator().annotate(img.copy(),results)
        annotated_frame = sv.BoxAnnotator().annotate(annotated_frame,results)
        annotated_frame = sv.LabelAnnotator().annotate(annotated_frame,results,labels)

        results_all.append({
            "filename":filename,
            "boxes":results.xyxy,
            "masks":masks_resized,
            "scores":scores,
            "classes":classes,
            "annotated_frame":annotated_frame

        })
        
    return results_all

if __name__ == "__main__":
    detect_model()