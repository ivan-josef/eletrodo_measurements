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


img_lateral = 'test_lateral-view/ct155_1.jpg'
resultados_eletrodo = []

result_desgaste = lat.desgaste(img_lateral) # [{'classe': [2], 'altura': np.float32(23.0)}]
classe = result_desgaste[0]['classe'][0]
altura_medida = result_desgaste[0]['altura']

altura_ref = ref.ref_por_classe[classe]['altura']

print(f'a diferença de alturas é {altura_ref - altura_medida}')

