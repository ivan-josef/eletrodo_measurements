import running_inference as ri
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def debug():
    while True:
        q = input('nugget,centralizacao, rugosidade ou rebarba?: ')
        if q.lower() == 'nugget':
            imgs = nugget()
            for img in imgs:
                resized = cv2.resize(img,(1920,1080),cv2.INTER_NEAREST)    

                cv2.imshow('mascaras',resized)
                cv2.imshow('bordas',cv2.Canny(resized,50,150))
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
        elif q.lower() == 'rebarba':
            imgs = rebarba()
            for img in imgs:
                resized = cv2.resize(img,(1920,1080),cv2.INTER_NEAREST)
                cv2.imshow('rebarbas',resized)
                cv2.waitKey(0)
        elif ord('q'):
            break
        else:
            continue
        cv2.destroyAllWindows()



def fit_circle_least_squares(contour):
    pts = contour.reshape(-1, 2)

    x = pts[:, 0]
    y = pts[:, 1]

    x_m = np.mean(x)
    y_m = np.mean(y)

    u = x - x_m
    v = y - y_m

    Suu = np.sum(u*u)
    Suv = np.sum(u*v)
    Svv = np.sum(v*v)
    Suuu = np.sum(u*u*u)
    Svvv = np.sum(v*v*v)
    Suvv = np.sum(u*v*v)
    Svuu = np.sum(v*u*u)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([(Suuu + Suvv)/2.0, (Svvv + Svuu)/2.0])

    uc, vc = np.linalg.solve(A, B)

    xc = x_m + uc
    yc = y_m + vc

    radius = np.mean(np.sqrt((x - xc)**2 + (y - yc)**2))

    return (int(xc), int(yc)), int(radius)



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

        print(f'{filename}')
        print(f"Grad mean: {grad_mean}")
        print(f"Grad std: {grad_std}")

        
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
    results_rebarba = []
    for result in results_all:
        masks = result['masks']
        filename = result['filename']
        
        zeros_mask = np.zeros(ref.shape,dtype=np.uint8)
        for i,mask in enumerate(masks):
            if i == 0:
                zeros_mask[mask] = (255,0,0)

        zeros_mask = cv2.GaussianBlur(zeros_mask, (5,5), 0)
        edges = cv2.Canny(zeros_mask,50,150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours,key=cv2.contourArea)
        pts = contour.reshape(-1, 2)
        (x_axis,y_axis),_ = fit_circle_least_squares(contour)
        distances = np.sqrt((pts[:,0] - x_axis)**2 + (pts[:,1] - y_axis)**2)
        threshold = np.percentile(distances, 80)
        filtered_pts = pts[distances < threshold]
        center, radius = fit_circle_least_squares(filtered_pts)

        # 1. Anel fino na posição radius+50
        ring_mask = np.zeros(ref.shape[:2], dtype=np.uint8)
        cv2.circle(ring_mask, center, radius + 50, 255, 8)  # espessura do anel

        # 2. Estima a cor do background (cantos da imagem, longe do objeto)
        h, w = ref.shape[:2]
        margin = 30
        corners = [
            ref[0:margin, 0:margin],
            ref[0:margin, w-margin:w],
            ref[h-margin:h, 0:margin],
            ref[h-margin:h, w-margin:w],
        ]
        bg_color = np.mean([c.mean(axis=(0,1)) for c in corners], axis=0)  # BGR médio

        # 3. Diferença de cor dos pixels no anel vs background
        ring_pixels_coords = np.where(ring_mask > 0)
        ring_pixels = ref[ring_pixels_coords].astype(np.float32)  # shape (N, 3)

        diff = np.linalg.norm(ring_pixels - bg_color, axis=1)  # distância euclidiana no espaço BGR

        # 4. Pixels muito diferentes do background = objeto = rebarba
        color_threshold = 30  # ajuste conforme a imagem
        burr_mask_flat = diff > color_threshold
        burr_count = burr_mask_flat.sum()
        has_burrs = burr_count > 10  # mínimo de pixels para evitar ruído

        # 5. Reconstrói máscara de rebarbas para visualização
        burr_viz = np.zeros(ref.shape[:2], dtype=np.uint8)
        ys, xs = ring_pixels_coords
        burr_viz[ys[burr_mask_flat], xs[burr_mask_flat]] = 255

        # Visualização
        draw_img = ref.copy()
        cv2.circle(draw_img, center, radius,      (0, 255, 0), 2)
        cv2.circle(draw_img, center, radius + 50, (0, 0, 255), 2)
        draw_img[burr_viz > 0] = (0, 255, 255)  # rebarba em amarelo

        print(f"Rebarbas: {'SIM' if has_burrs else 'NÃO'} ({burr_count} pixels no anel)")
        results_rebarba.append(draw_img)

    return results_rebarba






        


dataset = 'test'
img = 'test/ct1661778873155.0512855.jpg'
ref = cv2.imread(img)
model = 'modelos/rf-detr_model_top-view.pth'
results_all = ri.detect_model(model,img)

output = 'output'
os.makedirs(output,exist_ok=True)


debug()
cv2.destroyAllWindows()

        

