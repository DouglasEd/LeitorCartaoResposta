import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def CalcAng(Bordas):
    x1,y1= cv2.boundingRect(Bordas[0])[:2]
    x2,y2= cv2.boundingRect(Bordas[1])[:2]
    dx = x1-x2
    dy = y1-y2
    Rad= math.atan2(dx,dy)
    Grau = math.degrees(Rad)
    return Grau
def rotate(img, Canto, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    
    # Se o ponto de rotação não for especificado, usar o centro da imagem
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    
    # Obter a matriz de rotação
    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
    
    # Calcular o novo tamanho da imagem após a rotação para evitar corte
    cos = abs(rotMat[0, 0])
    sin = abs(rotMat[0, 1])

    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)

    # Ajustar a matriz de rotação para que a imagem inteira caiba no novo espaço
    rotMat[0, 2] += (new_width - width) // 2
    rotMat[1, 2] += (new_height - height) // 2

    # Realizar a rotação com o novo tamanho
    rotated_img = cv2.warpAffine(img, rotMat, (new_width, new_height))

    # Calcular as novas coordenadas do ponto (min_x, min_y) aplicando a matriz de rotação
    x, y, w, h = cv2.boundingRect(Canto)
    new_min_x, new_min_y = cv2.transform(np.array([[[x, y]]]), rotMat)[0][0]

    # Retornar a imagem rotacionada e as novas coordenadas
    return rotated_img, int(new_min_x), int(new_min_y)

def resize(img):
    x,y = img.shape[:2]
    proporcao = y/x
    largura = int(953*proporcao)
    redimencionado = cv2.resize(img,(largura,953),interpolation=cv2.INTER_LINEAR)
    return redimencionado
def CortarGabarito(img_path):
    # Carregar a imagem e converter para tons de cinza
    img = cv2.imread(img_path)
    img = resize(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Respostas={1:None,2:None,3:None,4:None,5:None,6:None,7:None,8:None,9:None,10:None,
               11:None,12:None,13:None,14:None,15:None,16:None,17:None,18:None,19:None,20:None,
               21:None,22:None,23:None,24:None,25:None,26:None,27:None,28:None,29:None,30:None,
               31:None,32:None,33:None,34:None,35:None,36:None,37:None,38:None,39:None,40:None,
               41:None,42:None,43:None,44:None,45:None,46:None,47:None,48:None,49:None,50:None,}
    Alts=['A','B','C','D','E']
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
            #cv2.rectangle(img,(x,y),(x+10,y+10),(0,255,0), thickness=2)

            max_y = max(max_y,y)
            min_y = min(min_y,y)
            max_x = max(max_x,x)
            min_x = min(min_x,x)

    Ang=CalcAng(Cantos)*-1
    img,min_x,min_y=rotate(img,Cantos[1],Ang)

    #cv2.rectangle(img,(min_x,min_y),(min_x+10,min_y+10),(255,0,0), thickness=2)

    if min_x < max_x and min_y < max_y:
        img = img[min_y+20:min_y+480,min_x+20:min_x+580]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area, max_area = 50, 200
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    tamx,tamy=(round(img.shape[1]/15),round(img.shape[0]/4))
    print(f'{tamx} {tamy}')
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = thresh[y:y+h, x:x+w]
        sub_regions = dividir_em_sub_regioes(roi)

        proporcoes = [cv2.countNonZero(sub_reg) / (sub_reg.shape[0] * sub_reg.shape[1]) for sub_reg in sub_regions]

        # Se a proporção máxima for maior que um determinado limiar, considerar a bolha como preenchida
        if np.max(proporcoes) > 0.3:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),thickness=2)
            quest=1+(math.floor(x/tamx))+(math.floor(y/tamy)*15)
            alt=int((y-(tamy*(math.floor(y/tamy))))/((tamy-30)/5))

            cv2.putText(img,f'{Alts[alt-1]}',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,.5,(255,0,0),thickness=1)
            Respostas[quest]=Alts[alt-1]
            #Formular para descobrir a questão 1+(math.floor(x/tamx))+(math.floor(y/tamy)*15)
    '''x1,y1=(img.shape[1],img.shape[0])
    print(f'{x1} {y1}')
    for i in range(0,15):
        for j in range(0,4):
            cv2.rectangle(img,(tamx*i,tamy*j),(tamx*(i+1),tamy*(j+1)),(255,0,0),thickness=2)
            cv2.rectangle(img,(tamx*i,tamy*j),(tamx*(i+1),tamy*j+30),(255,255,0),thickness=2)'''
    return img, Respostas
def ShowImg(img, Name='imagem'):
    plt.imshow(img)
    plt.show()

def CompararRespostas(Respostas):
    acertos=0
    gabarito = {
1: 'B', 2: 'A', 3: 'D', 4: 'A', 5: 'E', 6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'B',11: 'A', 12: 'C', 13: 'C', 14: 'E', 15: 'D',16: 'B', 17: 'E', 18: 'C', 19: 'A', 20: 'E', 21: 'D', 22: 'E', 23: 'E', 24: 'A', 25: 'C', 26: 'C', 27: 'D', 28: 'B', 29: 'D', 30: 'D', 31: 'A', 32: 'D', 33: 'D', 34: 'B', 35: 'C', 36: 'D', 37: 'B', 38: 'D', 39: 'D', 40: 'D', 41: 'D', 42: 'E', 43: 'C', 44: 'A', 45: 'D', 46: 'B', 47: 'C', 48: 'A', 49: 'D', 50: 'E' 
}
    for i in range(1,16):
        if Respostas[i] == gabarito[i]:
            acertos+=1

    return acertos