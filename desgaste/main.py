import cv2
import numpy as np

def calcular_desgaste_subtracao(caminho_imagem):
    # Fator calibrado (ajuste o 2162 se a peça de 23mm der outro valor)
    fator_mm_por_pixel = 23.0 / 2162.0 

    img = cv2.imread(caminho_imagem)
    if img is None:
        print("Erro ao carregar imagem.")
        return

    # 1. Separar os canais de cores primárias (BGR no OpenCV)
    b, g, r = cv2.split(img)

    # 2. O TRUQUE: Subtrair o Azul do Vermelho.
    # Cobre: Vermelho alto, Azul baixo = Resultado Positivo (Branco)
    # Fundo/Sombras: Vermelho e Azul iguais = Resultado Zero (Preto)
    # O cv2.subtract garante que valores negativos virem 0.
    img_cobre = cv2.subtract(r, b)

    # 3. Aplicar um desfoque forte para matar qualquer pixel de ruído perdido
    img_blur = cv2.GaussianBlur(img_cobre, (31, 31), 0)

    # 4. Binarizar: Tudo que for maior que 40 (cobre) vira 255. O resto morre.
    _, thresh = cv2.threshold(img_blur, 40, 255, cv2.THRESH_BINARY)

    # 5. O reflexo branco no topo da peça vai virar um "buraco" preto (pois branco tem Vermelho e Azul iguais).
    # Usamos um kernel moderado apenas para tampar esse buraco dentro da peça.
    kernel = np.ones((51, 51), np.uint8)
    mask_final = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 6. Procurar os contornos na máscara limpa
    contornos, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        print("Nenhum eletrodo encontrado.")
        return

    # 7. Pegar apenas a maior massa da imagem (o eletrodo)
    maior_contorno = max(contornos, key=cv2.contourArea)

    # 8. Extrair a caixa e calcular a altura
    x, y, w, h = cv2.boundingRect(maior_contorno)
    altura_mm = h * fator_mm_por_pixel

    # --- DESENHOS E VISUALIZAÇÃO ---
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 6)
    texto = f"Altura: {h}px ({altura_mm:.2f}mm)"
    cv2.putText(img, texto, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)

    # Janelas redimensionáveis
    cv2.namedWindow("1 - Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2 - Mascara de Subtracao R-B", cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow("1 - Original", 800, 600)
    cv2.resizeWindow("2 - Mascara de Subtracao R-B", 800, 600)

    cv2.imshow("1 - Original", img)
    cv2.imshow("2 - Mascara de Subtracao R-B", mask_final)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Rode com a sua imagem
calcular_desgaste_subtracao('desgaste/images/frame_1775765978.jpg')