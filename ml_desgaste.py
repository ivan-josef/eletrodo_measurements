import os
import cv2
import numpy as np
from PIL import Image
import supervision as sv
from rfdetr import RFDETRSegMedium

model = RFDETRSegMedium(pretrain_weights='rf-detr-segM-model.pth')
model.optimize_for_inference()

images = 'test'
output = 'output'
os.makedirs(output, exist_ok=True)
    


# imagem de calibração 

calib_path = 'calibration_img.jpg'
calib_img = cv2.imread(calib_path)
calib_img = cv2.cvtColor(calib_img,cv2.COLOR_BGR2RGB)
calib_pil = Image.fromarray(calib_img).resize((640,640))
H, W = calib_img.shape[:2]

calib = model.predict(calib_pil,threshold=0.5)
print(calib)

scale_x = W / 640
scale_y = H / 640

calib.xyxy[:, [0, 2]] *= scale_x
calib.xyxy[:, [1, 3]] *= scale_y

labels = [
        f"{class_id} {conf:.2f}"
        for class_id, conf in zip(calib.class_id, calib.confidence)
    ]

# annotated = sv.BoxAnnotator().annotate(calib_img, calib)
# annotated = sv.LabelAnnotator().annotate(annotated, calib, labels)



# for filename in os.listdir(images):

#     if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         continue

#     img_path = os.path.join(images, filename)

#     # imagem original
#     original = cv2.imread(img_path)
#     original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
#     H, W = original.shape[:2]

#     # imagem para inferência
#     img = Image.fromarray(original).resize((640, 640))

#     results = model.predict(img, threshold=0.5)

#     # 🔥 ESCALAR BBOX
#     scale_x = W / 640
#     scale_y = H / 640

#     results.xyxy[:, [0, 2]] *= scale_x
#     results.xyxy[:, [1, 3]] *= scale_y

#     # 🔥 ESCALAR MÁSCARAS
#     if results.mask is not None:
#         resized_masks = []
#         for mask in results.mask:
#             mask = mask.astype(np.uint8) * 255
#             mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
#             resized_masks.append(mask.astype(bool))

#         results.mask = np.array(resized_masks)

#     # labels
#     labels = [
#         f"{class_id} {conf:.2f}"
#         for class_id, conf in zip(results.class_id, results.confidence)
#     ]

#     # desenhar na ORIGINAL
#     annotated = sv.MaskAnnotator().annotate(original.copy(), results)
#     annotated = sv.BoxAnnotator().annotate(annotated, results)
#     annotated = sv.LabelAnnotator().annotate(annotated, results, labels)

#     output_path = os.path.join(output, filename)

#     cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))