import top_view_functions as top
import desgaste as lat
import running_inference as ri 
import dicionario_ref as ref
import cv2


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


# img_lateral = 'eletrodo_measurements/test_lateral-view/ct155_1.jpg'

# result_desgaste = lat.desgaste(img_lateral) # [{'classe': [2], 'altura': np.float32(23.0)}]
# print(result_desgaste)
# classe = result_desgaste[0]['classe'][0]
# print(classe)
# altura_medida = result_desgaste[0]['altura']
# print(altura_medida)

# altura_ref = ref.ref_por_classe[classe]['altura']
# desgaste = altura_ref - altura_medida

# print(f'o desgaste é {desgaste}')

img_tview = 'eletrodo_measurements/test_top-view/rugos.jpg'

obj_top_view = top.TopViewFun(img_tview)
# #nugget

# results_nugget,_ = obj_top_view.nugget()

# #centralizacao
# results_centralizacao,_ = obj_top_view.centralizacao()

# #rugosidade 
# results_rugosidade,_ = obj_top_view.rugosidade()

# #rebarba
# results_rebarba,_ = obj_top_view.detect_rebarba()

obj_top_view.debug()
