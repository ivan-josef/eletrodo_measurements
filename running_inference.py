import cv2
import supervision as sv
import os 
from rfdetr import RFDETRSegMedium



def separate_img(path):
    if os.path.isdir(path):
        images = []
        for f in os.listdir(path):
            file = os.path.join(path,f)  
            img = cv2.imread(file)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            images.append(img)
            return images
    else:
        img = cv2.imread(path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img
            
            
def detect_model(model, path):
    
    imgs = separate_img(path)

    modelo = RFDETRSegMedium(pretrain_weights=model)

    results_all = []
    for img in imgs:
        result = modelo.predict(img,threshold=0.5)
        results_all.append(result)
        return results_all 