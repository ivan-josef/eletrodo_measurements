import running_inference as ri
import os
import cv2
import numpy as np


dataset = 'test'
img = 'test/ct331778873285.549483.jpg'
model = 'rf-detr_model_top-view.pth'
results_all = ri.detect_model(model,img)
output = 'output'
os.makedirs(output,exist_ok=True)

def debug():
    imgs = nugget()
    for i,img in imgs:
        resized = cv2.resize(img,(1920,1080),cv2.INTER_NEAREST)    

        cv2.imshow('mascaras',resized)
        cv2.imshow('bordas',sobel_edge_detector(resized))
        cv2.waitKey(0)


def sobel_edge_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
    return grad_norm



def nugget():
    list_imgs = []
    for result in results_all:
        filename = result["filename"]
        annotated_frame = result["annotated_frame"]
        masks = result["masks"]
        
        zeros_mask = np.zeros(annotated_frame.shape,dtype=np.uint8)
        for i, mask in enumerate(masks):
            result_img = zeros_mask.copy()
            result_img[mask] = (255,0,0)
            list_imgs.append((i,result_img))
            ys,xs = np.where(mask)
            tamanho = len(ys) # quantidade de pixels na mascara -> nos da o desgaste 
            print(f'mascara {i} da imagem {filename} = {tamanho} pixels' )

            #cv2.circle(result_img,(int(xs.mean()),int(ys.mean())),5,(0,255,0,5))
            #output_path = os.path.join(output,filename)
            #cv2.imwrite(output_path,annotated_frame)


    return list_imgs

def centralização():
    for result in results_all:
        masks = result["masks"]
        annotated_frame = result["annotated_frame"]


        zeros_mak = np.zeros(annotated_frame.shape,dtype=np.uint8)

        for i,mask in enumerate(masks):
            if i == 0:
                zeros_mak[mask] = annotated_frame[mask]
                ys_0,xs_0 = np.where(mask)
            if i == 1:
                zeros_mak[mask] = annotated_frame[mask]
                ys_1,xs_1 = np.where(mask)

        cv2.circle(zeros_mak,(int(xs_0.mean()),int(ys_0.mean())),10,(0,0,0),5)
        cv2.circle(zeros_mak,(int(xs_1.mean()),int(ys_1.mean())),10,(255,255,255),5)
        cv2.imshow('teste',zeros_mak)
        cv2.waitKey(0)
        


centralização()
cv2.destroyAllWindows()

        

