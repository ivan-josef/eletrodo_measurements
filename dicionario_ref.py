import calibracao_desgaste as calibracao
# calibração

calib_img = 'calib_img.jpg'
resolution = calibracao.calib(calib_img)

# medidas de eletrodos
ref_por_classe = {
    2 : {'altura':23,'diametro':3.0,'distancia':25},
    3 : {'altura':23,'diametro':3.0,'outro_param':25},
    4 : {'altura':23,'diametro':3.0,'outro_param':25}
}
