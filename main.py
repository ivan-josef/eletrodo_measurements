import running_inference as ri
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def debug():
    q = input('nugget,centralizacao ou rugosidade?: ')
    if q.lower() == 'nugget':
        imgs = nugget()
        for img in imgs:
            resized = cv2.resize(img,(1920,1080),cv2.INTER_NEAREST)    

            cv2.imshow('mascaras',resized)
            cv2.imshow('bordas',sobel_edge_detector(resized))
            cv2.waitKey(0)
    elif q.lower() == 'centralizacao':
        imgs = centralizacao()
        for img in imgs:
            resized = cv2.resize(img,(1920,1080),cv2.INTER_NEAREST)    

            cv2.imshow('mascaras',resized)
            cv2.waitKey(0)
    elif q.lower() == 'rugosidade':
        dic = rugosidade()
        for k,v in dic.items():
            print(f'para {k} rugosidade é {v}')




def sobel_edge_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
    return grad_norm



def nugget():
    results_nugget = []
    for result in results_all:
        filename = result["filename"]
        masks = result["masks"]
        
        zeros_mask = np.zeros(ref.shape,dtype=np.uint8)
        for i, mask in enumerate(masks):
            result_img = zeros_mask.copy()
            result_img[mask] = (255,0,0)
            results_nugget.append(result_img)
            ys,xs = np.where(mask)
            tamanho = len(ys) # quantidade de pixels na mascara -> nos da o desgaste 
            print(f'mascara {i} da imagem {filename} = {tamanho} pixels' )

    
    return results_nugget

def centralizacao():
    results_centralizacao = []
    for result in results_all:
        masks = result["masks"]
        annotated_frame = result["annotated_frame"]


        zeros_mak = np.zeros(ref.shape,dtype=np.uint8)

        for i,mask in enumerate(masks):
            if i == 0:
                zeros_mak[mask] = annotated_frame[mask]
                ys_0,xs_0 = np.where(mask)
            if i == 1:
                zeros_mak[mask] = annotated_frame[mask]
                ys_1,xs_1 = np.where(mask)

        print(f'ponto 1: {xs_0.mean(),ys_0.mean()}')
        print(f'ponto 2: {xs_1.mean(),ys_1.mean()}')
        cv2.circle(zeros_mak,(int(xs_0.mean()),int(ys_0.mean())),5,(0,0,0),5)
        cv2.circle(zeros_mak,(int(xs_1.mean()),int(ys_1.mean())),5,(255,255,255),5)


        dist = math.dist((xs_0.mean(),ys_0.mean()),(xs_1.mean(),ys_1.mean()))
        print(f'a distancia euclidiana é: {dist}')

        results_centralizacao.append(zeros_mak)

    return results_centralizacao

def rugosidade():
    results_rugosidade = {}
    for result in results_all:
        filename = result["filename"]
        boxes = result["boxes"]

        maior_bb = boxes[0]
        x1,y1,x2,y2 = maior_bb
        meio_y = (y1+y2) // 2

        img_teste = os.path.join(dataset,filename)
        teste_img = cv2.imread(img_teste)
        
        gray = cv2.cvtColor(teste_img, cv2.COLOR_BGR2GRAY)
        gaus = cv2.GaussianBlur(gray,(31,31),0)

        linha_pixels = gaus[int(meio_y),int(x1):int(x2)]


        grad = np.abs(np.diff(linha_pixels))

        grad_mean = np.mean(grad)
        grad_std = np.std(grad)

        # print(f'{filename}')
        # print(f"Grad mean: {grad_mean}")
        # print(f"Grad std: {grad_std}")

        
        rugos = ''
        if grad_mean > 48:
            rugos = True
            results_rugosidade[filename] = rugos
        elif grad_mean < 38:
            rugos = False
            results_rugosidade[filename] = rugos
        else:
            rugos = None
            results_rugosidade[filename] = rugos
    
        plt.figure()
        plt.plot(linha_pixels)
        plt.title("Perfil de intensidade no eixo central")
        plt.xlabel("Posição (pixels)")
        plt.ylabel("Intensidade (0-255)")
        plt.savefig(f'{filename}')
        plt.close()

    return results_rugosidade


def rebarba():
    for result in results_all:
        masks = result['masks']
        filename = result['filename']
        
        zeros_mask = np.zeros(ref.shape,dtype=np.uint8)
        for i,mask in enumerate(masks):
            if i == 0:
                zeros_mask[mask] = (255,0,0)
        edges = cv2.Canny(zeros_mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours,key=cv2.contourArea)
        (x_axis,y_axis),radius = cv2.minEnclosingCircle(contour)
        center = (int(x_axis),int(y_axis))
        radius = int(radius)

        draw_img = np.zeros(ref.shape,dtype=np.uint8)
        cv2.drawContours(draw_img, [contour], -1, (255, 0, 0), 2)
        cv2.circle(ref,center,radius,(0,255,0),2)
        cv2.circle(ref,center,radius+50,(0,0,255),2)
        cv2.imshow('teste',ref)
        cv2.waitKey(0)






        


dataset = 'test'
img = 'test/ct1661778873155.0512855.jpg'
ref = cv2.imread(img)
model = 'rf-detr_model_top-view.pth'
results_all = ri.detect_model(model,img)

output = 'output'
os.makedirs(output,exist_ok=True)


rebarba()
cv2.destroyAllWindows()

        

