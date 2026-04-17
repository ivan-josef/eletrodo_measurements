import cv2 
import numpy as np 
import os

class Sub_Pixel_Edge():
    def __init__(self,path):
        self.path = path
        self.files = os.listdir(self.path)
        self.index = 0 
        self.window = 'eletrodo raw'

        # sobel params
        self.scale = 1 
        self.delta = 0 
        self.ddepth = cv2.CV_16S

        #trackbars
        self.setup_trackbars()

    def nothing(self,x):
        pass

    def edge_sobel(self, clahe_limit, bilat_color, bilat_space, thresh_val, k_size):
        # 1. Converter para o espaço de cores LAB
        lab = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2LAB)

        # 2. Separar os canais L, A, B
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 3. Criar o objeto CLAHE e aplicar apenas no canal L
        clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # 4. Juntar os canais novamente e voltar para BGR
        merged_lab = cv2.merge((cl, a_channel, b_channel))
        self.img_clahe = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        #filtros passa baixa
        #median = cv2.medianBlur(self.img_clahe,5)
        gaussian = cv2.bilateralFilter(self.img_clahe,9,bilat_color,bilat_space)

        #escala de cinza
        gray = cv2.cvtColor(gaussian,cv2.COLOR_BGR2GRAY)
        #imagem dividia em seus canais rgb
        b, g, r = cv2.split(gaussian)

        #calculo dos gradiantes em x e y
        self.grad_x = cv2.Sobel(r, self.ddepth, 1, 0, ksize=3, scale=self.scale, delta=self.delta, borderType=cv2.BORDER_DEFAULT)
        self.grad_y = cv2.Sobel(r, self.ddepth, 0, 1, ksize=3, scale=self.scale, delta=self.delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(self.grad_x)
        abs_grad_y = cv2.convertScaleAbs(self.grad_y)
        self.grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        #threshold
        self.ret, self.dst = cv2.threshold(self.grad,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #erosao/dilatação
        #kernel = np.ones((3,3),np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        self.dst_erode = cv2.erode(self.dst,kernel)
        self.dst_dilate = cv2.dilate(self.dst,kernel)

        #contornos
        



    def setup_trackbars(self):
        cv2.namedWindow('Controles')
        cv2.resizeWindow('Controles', 400, 250) # Tamanho da janelinha

        # Nome da Barra, Nome da Janela, Valor Inicial, Valor Máximo, Callback
        cv2.createTrackbar('CLAHE Limit', 'Controles', 150, 500, self.nothing) # Lembre de dividir por 10 depois!
        cv2.createTrackbar('Bilat Color', 'Controles', 75, 1000, self.nothing)
        cv2.createTrackbar('Bilat Space', 'Controles', 75, 1000, self.nothing)
        cv2.createTrackbar('Threshold', 'Controles', 50, 255, self.nothing)
        cv2.createTrackbar('Kernel Size', 'Controles', 3, 31, self.nothing)
     
        
        

    def run(self):
        while True:
            self.complete_path = os.path.join(self.path,self.files[self.index])
            self.img_raw = cv2.imread(self.complete_path)
            self.img_raw = cv2.resize(self.img_raw,(1280,720))

            # Lendo as trackbars da janela 'Controles'
            clahe_val = cv2.getTrackbarPos('CLAHE Limit', 'Controles') / 10.0 # O truque do float!
            b_color = cv2.getTrackbarPos('Bilat Color', 'Controles')
            b_space = cv2.getTrackbarPos('Bilat Space', 'Controles')
            thresh = cv2.getTrackbarPos('Threshold', 'Controles')
            k_val = cv2.getTrackbarPos('Kernel Size', 'Controles')

            # Truque para forçar o Kernel a ser SEMPRE ímpar e >= 1
            k_size = max(1, k_val if k_val % 2 != 0 else k_val + 1)

            # Agora passamos os valores lidos para o seu processamento:
            self.edge_sobel(clahe_val, b_color, b_space, thresh, k_size)

            cv2.imshow('sobel',self.grad)
            cv2.imshow('binario',self.dst_erode)
            cv2.imshow(self.window,self.img_clahe)
            self.key = cv2.waitKey(1) & 0xFF 
            if self.key == ord('d'):
                self.index += 1
            elif self.key == ord('a'):
                self.index -= 1
            elif self.key == ord('q'):
                break

            if self.index < 0:
                self.index = 0
            if self.index >= len(self.files) - 1:
                self.index = len(self.files) - 1


path = 'images'
obj = Sub_Pixel_Edge(path)
obj.run()
cv2.destroyAllWindows()
            