import numpy as np
import cv2

teste = np.zeros((300, 300, 3), dtype=np.uint8)
teste[:] = (0, 255, 0)

cv2.namedWindow("TESTE_INICIAL", cv2.WINDOW_NORMAL)
cv2.imshow("TESTE_INICIAL", teste)
cv2.waitKey(1)


from model_manager import manager
import top_view_functions as top
import desgaste as lat
import dicionario_ref as ref
import matplotlib.pyplot as plt
import calibracao_desgaste as calibracao


manager.warmup([
    'modelos/rf-detr-SEGM-lateral.pth',
    'modelos/rf-detr_model_top-view.pth'
])



#abrir a camera
#crop
#envia frame lateral-view pra inferencia
    # identifica o tipo de eletrodo
    #envia informações de referencia do eletrodo
#troca de camera
#crop
#envia frame top-view pra inferencia
#maquina de estados
#relatorio

# lateral view

# frames = [lateral,superior]




img_lateral = '/home/ivan/Documentos/eletrode_raw/frames/eletrodos_caixa_nova_rastreados/ct155/38_lat.jpg'
img_tview = '/home/ivan/Documentos/eletrode_raw/frames/eletrodos_caixa_nova_rastreados/ct155/38.jpg'

resolution = calibracao.calib('calib_img.jpg')


def main():


    resultados = {}

    result_desgaste,_ = lat.desgaste(img_lateral,resolution) # [{'classe': [2], 'altura': np.float32(23.0)}]
    classe = result_desgaste[0]['classe']
    altura_medida = result_desgaste[0]['altura']

    altura_ref = ref.ref_por_classe[classe]['altura']
    desgaste = altura_ref - altura_medida

    # top view

    obj_top_view = top.TopViewFun(img_tview)
    #nugget

    results_nugget,_ = obj_top_view.nugget()

    #centralizacao
    results_centralizacao,_ = obj_top_view.centralizacao()
    #rugosidade     
    results_rugosidade,_ = obj_top_view.rugosidade()

    # rebarba
    results_rebarba,is_poligonal,_,_ = obj_top_view.detect_rebarba()

    resultados['Desgaste: '] = desgaste
    resultados['Diâmetro do nugget: '] = results_nugget 
    resultados['Descentralização: '] = results_centralizacao
    resultados['Rugosidade: '] = results_rugosidade
    resultados['Rebarba: '] = results_rebarba
    resultados['Poligonal: '] = is_poligonal

    print(resultados)

    manager.shutdown()

def debug():

    result_desgaste,desgaste_mask = lat.desgaste(img_lateral,resolution,annotate=True) # [{'classe': [2], 'altura': np.float32(23.0)}]

    
    obj_top_view = top.TopViewFun(img_tview,annotate=True)
    results_nugget,nugget_mask = obj_top_view.nugget()
    print(nugget_mask.dtype)

    #centralizacao
    results_centralizacao,centralizacao_mask = obj_top_view.centralizacao()

    #rugosidade     
    results_rugosidade,rugos_mask = obj_top_view.rugosidade()
    rugosidade_val = obj_top_view.grad_mean
    

    # rebarba
    results_rebarba,is_poligonal,outlier_ratio,rebarb_mask = obj_top_view.detect_rebarba()  

    print(f'''
Desgaste: {result_desgaste}
Nugget: {results_nugget} 
Descentralizacao: {results_centralizacao}  
Rugosidade: {results_rugosidade}
Valor de gradiente: {rugosidade_val}
Poligonal: {is_poligonal}
Outliers: {outlier_ratio}
Rebarba: {results_rebarba} 
    ''')

    nugget_debug = cv2.resize(nugget_mask,(1920,1080))
    centralizacao_debug = cv2.resize(centralizacao_mask,(1920,1080))
    rebarba_debug = cv2.resize(rebarb_mask,(1920,1080))
    desgaste_debug = cv2.resize(desgaste_mask,(1920,1080))

    cv2.imshow('nugget',nugget_debug)
    cv2.imshow('centralizacao',centralizacao_debug)
    cv2.imshow('rebarb',rebarba_debug)    
    cv2.imshow('desgaste',desgaste_debug)      
  
    cv2.waitKey(0)  

    plt.figure()
    plt.plot(rugos_mask)
    plt.title("Perfil de intensidade no eixo central")
    plt.xlabel("Posição (pixels)")
    plt.ylabel("Intensidade (0-255)")
    plt.show()
    plt.close()


if __name__ == "__main__":
    debug()
