import cv2
import supervision as sv
import os 
from PIL import Image
import numpy as np 
import model_manager as manager


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
                        

def detect_model(model_path, path, resolution = 1296,threshold = 0.5,annotate = False):
    
    data = separate_img(path)

    results_all = []

    for filename,img in data:
        H,W = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((640,640))
        del img_rgb

        results = manager.predict(pil_img,threshold=threshold)
        del pil_img

        scale_y = H / 640
        scale_x = W / 640

        if results.xyxy is not None:        
            boxes = results.xyxy.copy()

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        
        masks_resized = []
        if results.mask is not None:
            for mask in results.mask:
                m = mask.astype("uint8") * 255

                m = cv2.resize(
                    m,
                    (W, H),
                    interpolation=cv2.INTER_NEAREST
                )

                masks_resized.append(m.astype(bool))

        scores = results.confidence.tolist() if results.confidence is not None else []
        classes = results.class_id.tolist() if results.class_id is not None else []

        # inference (só para debug)

        annotated_frame = None
        if annotate:
            det = sv.Detections(
                xyxy=boxes if boxes is not None else np.empty((0, 4)),
                mask=np.array(masks_resized) if masks_resized else None,
                confidence=np.array(scores),
                class_id=np.array(classes),
            )
            labels = [f"{c} {s:.2f}" for c, s in zip(classes, scores)]
            af = img.copy()
            af = sv.MaskAnnotator().annotate(af, det)
            af = sv.BoxAnnotator().annotate(af, det)
            af = sv.LabelAnnotator().annotate(af, det, labels)
            annotated_frame = af

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