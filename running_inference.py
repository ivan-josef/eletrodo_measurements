import cv2
import supervision as sv
import os 
from PIL import Image
import numpy as np 
from model_manager import manager
 

def detect_model(frame,threshold = 0.5,annotate = False):
    results_all = []
    H,W = frame.shape[:2]

    input_size = 640

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize((input_size,input_size))
    del img_rgb

    results = manager.predict(pil_img,threshold=threshold)
    del pil_img

    scale_y = H / input_size
    scale_x = W / input_size

    boxes = None
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
        af = frame.copy()
        af = sv.MaskAnnotator().annotate(af, det)
        af = sv.BoxAnnotator().annotate(af, det)
        af = sv.LabelAnnotator().annotate(af, det, labels)
        annotated_frame = af

    results_all.append({
        "boxes":boxes,
        "masks":masks_resized,
        "scores":scores,
        "classes":classes,
        "annotated_frame":annotated_frame

    })
        
    return results_all

if __name__ == "__main__":
    detect_model()