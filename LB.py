import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def CalcAng(Bordas):
    x1,y1,w1,h1 = cv2.boundingRect(Bordas[0])
    x2,y2,w2,h2 = cv2.boundingRect(Bordas[1])
    dx = x1-x2
    dy = y1-y2
    Rad= math.atan2(dx,dy)
    Grau = math.degrees(Rad)
    print(Grau)
    return Grau
def rotate(img,Canto,angle, rotPoint=None):
    (height,width) = img.shape[:2]
    x, y, h, w = cv2.boundingRect(Canto)
    if rotPoint == None:
        rotPoint = (width//2, height//2)
    x0,y0=rotPoint

    xn=x0+(x-x0)*math.cos(math.radians(angle)) + (y-y0) * math.sin(math.radians(angle))
    yn=y0+(x-x0)*math.sin(math.radians(angle)) + (y-y0) * math.cos(math.radians(angle))
    print(f'x0={x0} y0 = {y0} x={x} y={y} xn = {xn} yn = {yn} ang={math.radians(angle)}')

    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
    dim = (width,height)

    return cv2.warpAffine(img, rotMat, dim), int(xn), int(yn)

def CortarGabarito(img_path):
    # Carregar a imagem e converter para tons de cinza
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar filtro de limiarização
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    Cantos=[None,None]
    # Encontrar contornos e filtrar por tamanho
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area, max_area = 50, 200
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    max_y,max_x = 0,0
    min_y, min_x = float('inf'),float('inf')

    # Função para dividir a região da bolha em sub-regiões
    def dividir_em_sub_regioes(roi):
        height, width = roi.shape
        sub_region_height = height // 2
        sub_region_width = width // 2
        return [
            roi[:sub_region_height, :sub_region_width],
            roi[:sub_region_height, sub_region_width:],
            roi[sub_region_height:, :sub_region_width],
            roi[sub_region_height:, sub_region_width:]
        ]

    # Analisar cada contorno (bolha)
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = thresh[y:y+h, x:x+w]
        sub_regions = dividir_em_sub_regioes(roi)

        proporcoes = [cv2.countNonZero(sub_reg) / (sub_reg.shape[0] * sub_reg.shape[1]) for sub_reg in sub_regions]

        # Se a proporção máxima for maior que um determinado limiar, considerar a bolha como preenchida
        if np.max(proporcoes) > 0.7:
            if y < min_y and x < min_x:
                Cantos[0]=cnt
            elif y < min_y and x < max_x:
                Cantos[1]=cnt
            cv2.rectangle(img,(x,y),(x+10,y+10),(0,255,0), thickness=2)

            max_y = max(max_y,y)
            min_y = min(min_y,y)
            max_x = max(max_x,x)
            min_x = min(min_x,x)
    print(Cantos)
    print(f'{min_x}, {min_y} , {max_x}, {max_y}')

    Ang=CalcAng(Cantos)*-1
    img,min_x,min_y=rotate(img,Cantos[1],Ang)
    print(f'{min_x}, {min_y}')

    cv2.rectangle(img,(min_x,min_y),(min_x+10,min_y+10),(255,0,0), thickness=2)

    '''if min_x < max_x and min_y < max_y:
        img = img[min_y+20:min_y+480,min_x+20:min_x+580]'''
    return img
def ShowImg(img, Name='imagem'):
    plt.imshow(img)
    plt.show()