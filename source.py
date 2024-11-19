import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math


def lerPasta():
    if os.path.isdir('Imagens'):
        extensoes = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.ppm', '.pgm', '.pbm', '.ras', '.sr', '.exr']
        arquivos = os.listdir('Imagens')
        arquivosCompativeis=[]
        for arquivo in arquivos:
             for ext in extensoes:
                if os.path.isfile(os.path.join('Imagens', arquivo)) and arquivo.lower().endswith(ext):
                    arquivosCompativeis.append(arquivo)
        return arquivosCompativeis
    else:
        print("A pasta 'Imagens' não existe.")
def calcularAngulo(Bordas):
    x1,y1= cv2.boundingRect(Bordas[0])[:2]
    x2,y2= cv2.boundingRect(Bordas[1])[:2]
    dx = x1-x2
    dy = y1-y2
    rad= math.atan2(dx,dy)
    grau = math.degrees(rad)
    return grau
def rotacionarImagem(img, canto, angulo, ponto_de_rotacao=None):
    (altura, largura) = img.shape[:2]
    
    if ponto_de_rotacao is None:
        ponto_de_rotacao = (largura // 2, altura // 2)

    matriz_de_rotacao = cv2.getRotationMatrix2D(ponto_de_rotacao, angulo, 1.0)
    
    cos = abs(matriz_de_rotacao[0, 0])
    sen = abs(matriz_de_rotacao[0, 1])

    nova_largura = int(altura * sen + largura * cos)
    nova_altura = int(altura * cos + largura * sen)

    matriz_de_rotacao[0, 2] += (nova_largura - largura) // 2
    matriz_de_rotacao[1, 2] += (nova_altura - altura) // 2

    imagem_rotacionada = cv2.warpAffine(img, matriz_de_rotacao, (nova_largura, nova_altura))

    min_x, min_y, w, h = cv2.boundingRect(canto)
    novo_min_x, novo_min_y = cv2.transform(np.array([[[min_x, min_y]]]), matriz_de_rotacao)[0][0]

    return imagem_rotacionada, int(novo_min_x), int(novo_min_y)

def redimensionarImagem(img):
    x,y = img.shape[:2]
    proporcao = y/x
    largura = int(953*proporcao)
    img_redimensionada = cv2.resize(img,(largura,953),interpolation=cv2.INTER_LINEAR)
    return img_redimensionada

def cortarGabarito(img_path):
    img = cv2.imread(img_path)
    img = redimensionarImagem(img)
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Inicializar respostas com valores em branco
    respostas = {i: '' for i in range(1, 51)}
    alternativas = ['A', 'B', 'C', 'D', 'E']

    # Binarizar a imagem para detectar contornos
    img_cinza_tresh = cv2.threshold(img_cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cantos = [None, None]

    # Encontrar contornos
    contornos, _ = cv2.findContours(img_cinza_tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_minima, area_maxima = 50, 200
    contornos_filtrados = [cnt for cnt in contornos if area_minima < cv2.contourArea(cnt) < area_maxima]

    # Definir variáveis para limites de detecção
    max_y, max_x = 0, 0
    min_y, min_x = float('inf'), float('inf')

    def dividirEmSubRegioes(roi):
        altura, largura = roi.shape
        altura_da_sub_regiao = altura // 2
        largura_da_sub_regiao = largura // 2
        return [
            roi[:altura_da_sub_regiao, :largura_da_sub_regiao],
            roi[:altura_da_sub_regiao, largura_da_sub_regiao:],
            roi[altura_da_sub_regiao:, :largura_da_sub_regiao],
            roi[altura_da_sub_regiao:, largura_da_sub_regiao:]
        ]

    # Encontrar cantos da imagem de marcação
    for cnt in contornos_filtrados:
        x, y, largura, altura = cv2.boundingRect(cnt)
        roi = img_cinza_tresh[y:y+altura, x:x+largura]
        sub_regioes = dividirEmSubRegioes(roi)

        proporcoes = [cv2.countNonZero(sub_reg) / (sub_reg.shape[0] * sub_reg.shape[1]) for sub_reg in sub_regioes]

        if np.max(proporcoes) > 0.7:
            if y < min_y and x < min_x:
                cantos[0] = cnt
            elif y < min_y and x < max_x:
                cantos[1] = cnt

            max_y = max(max_y, y)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            min_x = min(min_x, x)

    angulo = calcularAngulo(cantos) * -1
    img, min_x, min_y = rotacionarImagem(img, cantos[1], angulo)

    if min_x < max_x and min_y < max_y:
        img = img[min_y+20:min_y+480, min_x+20:min_x+580]

    # Processar a imagem rotacionada e cortada
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cinza_tresh = cv2.threshold(img_cinza, 127, 255, cv2.THRESH_BINARY_INV)[1]
    contornos, _ = cv2.findContours(img_cinza_tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos_filtrados = [cnt for cnt in contornos if area_minima < cv2.contourArea(cnt) < area_maxima]
    tamanho_x, tamanho_y = (round(img.shape[1] / 15), round(img.shape[0] / 4))

    for cnt in contornos_filtrados:
        x, y, largura, altura = cv2.boundingRect(cnt)
        roi = img_cinza_tresh[y:y+altura, x:x+largura]
        sub_regioes = dividirEmSubRegioes(roi)

        proporcoes = [cv2.countNonZero(sub_reg) / (sub_reg.shape[0] * sub_reg.shape[1]) for sub_reg in sub_regioes]

        if np.max(proporcoes) > 0.3:
            cv2.rectangle(img, (x, y), (x + largura, y + altura), (255, 255, 0), thickness=2)
            questao = 1 + (math.floor(x / tamanho_x)) + (math.floor(y / tamanho_y) * 15)
            alternativa = int((y - (tamanho_y * (math.floor(y / tamanho_y)))) / ((tamanho_y - 30) / 5))

            cv2.putText(img, f'{alternativas[alternativa - 1]}', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 0, 0), thickness=1)
            
            # Atualizar a resposta da questão com a alternativa marcada
            if respostas[questao] == '':
                respostas[questao] = alternativas[alternativa - 1]
            else:
                respostas[questao] = 'Anulada'
            
    for questao in respostas:
        if respostas[questao] == '':
            respostas[questao] = "Vazio"
    return img, respostas

def mostrarImagem(img, Name='imagem'):
    plt.imshow(img)
    plt.show()

def compararRespostas(Respostas):
    acertos=0
    gabarito = {1: 'B', 2: 'A', 3: 'D', 4: 'A', 5: 'E', 6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'B',
                11: 'A', 12: 'C', 13: 'C', 14: 'E', 15: 'D',16: 'B', 17: 'E', 18: 'C', 19: 'A', 20: 'E', 
                21: 'D', 22: 'E', 23: 'E', 24: 'A', 25: 'C', 26: 'C', 27: 'D', 28: 'B', 29: 'D', 30: 'D', 
                31: 'A', 32: 'D', 33: 'D', 34: 'B', 35: 'C', 36: 'D', 37: 'B', 38: 'D', 39: 'D', 40: 'D', 
                41: 'D', 42: 'E', 43: 'C', 44: 'A', 45: 'D', 46: 'B', 47: 'C', 48: 'A', 49: 'D', 50: 'E' 
            }
    for i in range(1,16):
        if Respostas[i] == gabarito[i]:
            acertos+=1
    return acertos