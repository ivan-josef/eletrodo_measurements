import top_view_functions as top
import desgaste as lat
import dicionario_ref as ref
import cv2
from model_manager import manager

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
manager.use('modelos/rf-detr-SEGM-lateral.pth')

img_lateral = 'test_lateral-view/ct155_1.jpg'

result_desgaste = lat.desgaste(img_lateral) # [{'classe': [2], 'altura': np.float32(23.0)}]
print(result_desgaste)
classe = result_desgaste[0]['classe'][0]
print(classe)
altura_medida = result_desgaste[0]['altura']
print(altura_medida)

altura_ref = ref.ref_por_classe[classe]['altura']
desgaste = altura_ref - altura_medida

print(f'o desgaste é {desgaste}')


manager.use('modelos/rf-detr_model_top-view.pth')

img_tview = 'test_top-view/rugos.jpg'


# top view

obj_top_view = top.TopViewFun(img_tview,annotate=True)
#nugget

results_nugget,_ = obj_top_view.nugget()

#centralizacao
results_centralizacao,_ = obj_top_view.centralizacao()

#rugosidade 
results_rugosidade,_ = obj_top_view.rugosidade()

# rebarba
results_rebarba,_ = obj_top_view.detect_rebarba()

obj_top_view.debug()
cv2.destroyAllWindows()

manager.shutdown()




