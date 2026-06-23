import running_inference as ri
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt



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




class TopViewFun():
    def __init__(self,frame):
        self.model = 'eletrodo_measurements/modelos/rf-detr_model_top-view.pth'
        self.ref = cv2.imread(frame)
        results_all = ri.detect_model(self.model,frame)

        self.zeros_mask = np.zeros(self.ref.shape,dtype=np.uint8)

        for result in results_all:
            self.filename = result["filename"]
            self.masks = result["masks"]
            self.boxes = result["boxes"]
            self.scores = result["scores"]
            self.classes = result["classes"]
            self.annotated_frame = result["annotated_frame"]


    def nugget(self):
        results_nugget = []
        results_mask = []
    
        np_mask = self.zeros_mask.copy()
        for i, mask in enumerate(self.masks):
            result_img = np_mask.copy()
            if i == 1:
                result_img[mask] = (255,0,0)
            ys,xs = np.where(mask)
            size = len(ys) 
            results_nugget.append({i:size})
            results_mask.append(result_img)
            
        return results_nugget,results_mask
    
    def centralizacao(self):
        results_mask = []
        
        np_mask = self.zeros_mask.copy()
        for i,mask in enumerate(self.masks):
            if i == 0:
                np_mask[mask] = self.annotated_frame[mask]
                ys_0,xs_0 = np.where(mask)
            if i == 1:
                np_mask[mask] = self.annotated_frame[mask]
                ys_1,xs_1 = np.where(mask)


        cv2.circle(np_mask,(int(xs_0.mean()),int(ys_0.mean())),5,(0,0,0),5)
        cv2.circle(np_mask,(int(xs_1.mean()),int(ys_1.mean())),5,(255,255,255),5)


        dist = math.dist((xs_0.mean(),ys_0.mean()),(xs_1.mean(),ys_1.mean()))

        results_mask.append(np_mask)

        return dist,np_mask
    
    def rugosidade(self):

        results_rugosidade = {}
        
        maior_bb = self.boxes[0]
        x1,y1,x2,y2 = maior_bb
        meio_y = (y1+y2) // 2

        gray = cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)
        gaus = cv2.GaussianBlur(gray,(31,31),0)

        linha_pixels = gaus[int(meio_y),int(x1):int(x2)]

        
        grad = np.abs(np.diff(linha_pixels))

        grad_mean = np.mean(grad)
        grad_std = np.std(grad)

        
        rugos = ''
        if grad_mean > 48:
            rugos = True
            results_rugosidade[self.filename] = rugos
        elif grad_mean < 38:
            rugos = False
            results_rugosidade[self.filename] = rugos
        else:
            rugos = None
            results_rugosidade[self.filename] = rugos

        return results_rugosidade, linha_pixels
    
    def detect_rebarba(self):
        np_mask = self.zeros_mask.copy()
        for i, mask in enumerate(self.masks):
            if i == 0:
                np_mask[mask] = (255, 0, 0)

        np_mask = cv2.GaussianBlur(np_mask, (5, 5), 0)
        edges = cv2.Canny(np_mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)

        pts = contour.reshape(-1, 2)

        center_rough, _ = fit_circle_least_squares(contour)
        cx, cy = center_rough
        distances = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        threshold = np.percentile(distances, 80)
        filtered_pts = pts[distances < threshold]

        center, radius = fit_circle_least_squares(filtered_pts)

        # Anel de referência do background: radius+80 a radius+120
        # Longe do eletrodo e longe de qualquer rebarba possível
        bg_outer = np.zeros(self.ref.shape[:2], dtype=np.uint8)
        bg_inner = np.zeros(self.ref.shape[:2], dtype=np.uint8)
        cv2.circle(bg_outer, center, radius + 120, 255, -1)
        cv2.circle(bg_inner, center, radius + 80,  255, -1)
        bg_ring = cv2.subtract(bg_outer, bg_inner)

        # Cor do background LOCAL (próximo ao anel, mesmo iluminação)
        bg_coords = np.where(bg_ring > 0)
        bg_pixels = self.ref[bg_coords].astype(np.float32)
        bg_color = bg_pixels.mean(axis=0)  # BGR médio local

        # Anel de verificação: radius+50 com espessura 8px
        ring_mask = np.zeros(self.ref.shape[:2], dtype=np.uint8)
        cv2.circle(ring_mask, center, radius + 50, 255, 8)

        ring_coords = np.where(ring_mask > 0)
        ring_pixels = self.ref[ring_coords].astype(np.float32)
        diff = np.linalg.norm(ring_pixels - bg_color, axis=1)

        color_threshold = 30
        burr_mask_flat = diff > color_threshold
        burr_count = burr_mask_flat.sum()
        has_burrs = burr_count > 10

        # Visualização
        burr_viz = np.zeros(self.ref.shape[:2], dtype=np.uint8)
        ys, xs = ring_coords
        burr_viz[ys[burr_mask_flat], xs[burr_mask_flat]] = 255

        draw_img = self.ref.copy()
        cv2.circle(draw_img, center, radius,      (0, 255, 0), 2)
        cv2.circle(draw_img, center, radius + 50, (0, 0, 255), 2)
        draw_img[burr_viz > 0] = (0, 255, 255)

        return has_burrs, draw_img


    def debug(self):
        nugget_value,nugget_mask = self.nugget()
        centr_value,centr_mask = self.centralizacao()
        rugos_value,rugos_graf = self.rugosidade()
        rebarb_value,rebarb_mask = self.detect_rebarba()

        print(f'foram medidos {nugget_value} para o nugget, {centr_value} para a distância entre \
              centros, {rugos_value} para rugosidade e {rebarb_value} para rebarba')
        
        plt.figure()
        plt.plot(rugos_graf)
        plt.title("Perfil de intensidade no eixo central")
        plt.xlabel("Posição (pixels)")
        plt.ylabel("Intensidade (0-255)")
        plt.show()
        plt.close()

        nugget_debug = cv2.resize(nugget_mask,(1920,1080))
        centralizacao_debug = cv2.resize(centr_mask,(1920,1080))
        rebarba_debug = cv2.resize(rebarb_mask,(1920,1080))

        cv2.imshow('nugget',nugget_debug)
        cv2.imshow('centralizacao',centralizacao_debug)
        cv2.imshow('rebarb',rebarba_debug)
        



        


 

if __name__ == "__main__":
    pass