import running_inference as ri
import cv2
import numpy as np
import math
from model_manager import manager



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

    def __init__(self,frame,annotate = False, crop_box = None):


        img = cv2.imread(frame)

        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada: {frame}") 
        
        if crop_box is not None:
            x1,y1,x2,y2 = crop_box
            img = img[y1:y2,x1:x2]
        
        self.ref = img
        self._shape = img.shape
        self.zeros_mask = np.zeros(img.shape,dtype=np.uint8)

        manager.use('modelos/rf-detr_model_top-view.pth')

        
        results_all = ri.detect_model(

            img, annotate=annotate
        )

        result = results_all[0] 
        self.masks = result["masks"]           # list[np.ndarray bool H×W]
        self.boxes = result["boxes"]           # np.ndarray (N,4) | None
        self.scores = result["scores"]
        self.classes = result["classes"]
        # annotated_frame só existe se annotate=True
        self._annotated = result["annotated_frame"]

        self.detections = {}

        for i, classe in enumerate(self.classes):
            classe = int(classe)
            score = float(self.scores[i])

            if (
                classe not in self.detections
                or score > self.detections[classe]["score"]
            ):
                self.detections[classe] = {
                    "mask": self.masks[i],
                    "box": self.boxes[i],
                    "score": score,
                }

        ELECTRODE_CLASS_ID = 1  # CE
        NUGGET_CLASS_ID = 2     # CI

        if ELECTRODE_CLASS_ID not in self.detections:
            raise ValueError("Classe 2 (CE/eletrodo) não detectada.")

        if NUGGET_CLASS_ID not in self.detections:
            raise ValueError("Classe 3 (CI/nugget) não detectada.")

        self.electrode_mask = self.detections[ELECTRODE_CLASS_ID]["mask"]
        self.electrode_box = self.detections[ELECTRODE_CLASS_ID]["box"]

        self.nugget_mask = self.detections[NUGGET_CLASS_ID]["mask"]
        self.nugget_box = self.detections[NUGGET_CLASS_ID]["box"]

    def nugget(self):
    
        mask = self.nugget_mask
        ys,_ = np.where(mask)
        size = len(ys) 

        ref_copy = self.ref.copy()
        print(ref_copy.dtype)
        if self._annotated is not None:
            ref_copy[mask] = self._annotated[mask]
            
        return size, ref_copy
    
    def centralizacao(self):

        np_mask = self.zeros_mask.copy()
        mask0 = self.electrode_mask
        mask1 = self.nugget_mask
        if mask0 is None or mask1 is None:
            return None,None
        
        ys0,xs0 = np.where(mask0)
        ys1,xs1 = np.where(mask1)

        cx0,cy0 = xs0.mean(),ys0.mean()
        cx1,cy1 = xs1.mean(),ys1.mean()


        dist = math.dist((cx0,cy0),(cx1,cy1))

        if self._annotated is not None:
            np_mask[mask0] = self._annotated[mask0]
            np_mask[mask1] = self._annotated[mask1]

        cv2.circle(np_mask,(int(cx0),int(cy0)),5,(0,0,0),5)
        cv2.circle(np_mask,(int(cx1),int(cy1)),5,(255,255,255),5)

        return dist,np_mask
    
    def rugosidade(self):

    
        x1,y1,x2,y2 = self.electrode_box
        meio_y = int((y1+y2) // 2)

        gray = cv2.cvtColor(self.ref, cv2.COLOR_BGR2GRAY)
        gaus = cv2.GaussianBlur(gray,(31,31),0)
        del gray

        linha_pixels = gaus[meio_y,int(x1):int(x2)]
        del gaus

        
        grad = np.abs(np.diff(linha_pixels))
        self.grad_mean = np.mean(grad)
        
        if self.grad_mean > 48:
            rugos = True
        elif self.grad_mean < 38:
            rugos = False
        else:
            rugos = None

        return rugos, linha_pixels
    
    def detect_rebarba(self):

        mask0 = self.electrode_mask
        np_mask = self.zeros_mask.copy()
 
        np_mask[mask0] = 255

        np_mask = cv2.GaussianBlur(np_mask, (5, 5), 0)
        edges = cv2.Canny(np_mask, 50, 150)
        del np_mask
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        del edges

        contour = max(contours, key=cv2.contourArea)
        pts = contour.reshape(-1, 2)

        center_rough, _ = fit_circle_least_squares(pts)
        cx, cy = center_rough
        distances = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)

        threshold = np.percentile(distances, 80)
        filtered_pts = pts[distances < threshold]

        center, radius = fit_circle_least_squares(filtered_pts)

        # --- Detecção de poligonal (aproveita pts, center, radius já calculados) ---
        cx, cy = center
        all_distances = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        deviation = np.abs(all_distances - radius)
        outlier_mask = deviation > 15
        outlier_ratio = outlier_mask.sum() / len(pts)
        is_polygonal = outlier_ratio > 0.25

        H, W = self._shape[:2]

        # Anel de referência do background: radius+80 a radius+120
        # Longe do eletrodo e longe de qualquer rebarba possível
        bg_outer = np.zeros((H,W), dtype=np.uint8)
        bg_inner = np.zeros((H,W), dtype=np.uint8)
        cv2.circle(bg_outer, center, radius + 120, 255, -1)
        cv2.circle(bg_inner, center, radius + 80,  255, -1)
        bg_ring = cv2.subtract(bg_outer, bg_inner)
        del bg_outer,bg_inner

        # Cor do background LOCAL (próximo ao anel, mesmo iluminação)
        bg_coords = np.where(bg_ring > 0)
        del bg_ring
        bg_pixels = self.ref[bg_coords].astype(np.float32)
        bg_color = bg_pixels.mean(axis=0)  # BGR médio local

        # Anel de verificação: radius+50 com espessura 8px
        ring_mask = np.zeros((H,W), dtype=np.uint8)
        cv2.circle(ring_mask, center, radius + 50, 255, 8)
        ring_coords = np.where(ring_mask > 0)
        del ring_mask 

        ring_pixels = self.ref[ring_coords].astype(np.float32)
        diff = np.linalg.norm(ring_pixels - bg_color, axis=1)
        del ring_pixels

        color_threshold = 30
        burr_mask_flat = diff > color_threshold
        burr_count = burr_mask_flat.sum()
        has_burrs = burr_count > 10

        # Visualização
        draw_img = self.ref.copy()
        cv2.circle(draw_img, center, radius,       (0, 255, 0), 2)
        cv2.circle(draw_img, center, radius + 50,  (0, 0, 255), 2)
        ys, xs = ring_coords
        draw_img[ys[burr_mask_flat], xs[burr_mask_flat]] = (0, 255, 255)


        # Outliers do contorno em vermelho (poligonal)
        for pt, is_out in zip(pts[outlier_mask], [True]*outlier_mask.sum()):
            cv2.circle(draw_img, tuple(pt), 2, (0, 0, 255), -1)

        return has_burrs, is_polygonal, outlier_ratio, draw_img


    
if __name__ == "__main__":
    pass