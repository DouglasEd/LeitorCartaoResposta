import cv2
import numpy as np

def identificar_alternativas(img_path):
    """
    Identifica as alternativas marcadas em um cartão-resposta e exibe uma área de corte que engloba todas as áreas detectadas.

    Args:
        img_path (str): Caminho para a imagem do cartão-resposta.
    """
    # Carregar a imagem e converter para tons de cinza
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar filtro de limiarização
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Encontrar contornos e filtrar por tamanho
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area, max_area = 15, 20
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    # Variáveis para armazenar as coordenadas mínimas e máximas dos contornos
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0

    # Analisar cada contorno (bolha) e atualizar os limites de recorte
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

        cv2.rectangle(img,(x,y), (x+w,y+h),(0,255,0),2)

    # Verificar se encontramos contornos válidos
    if x_min < x_max and y_min < y_max:
        detected_area = img[y_min:y_max, x_min:x_max]
        cv2.imshow('Área Detectada', detected_area)
    else:
        print("Nenhuma área detectada para recorte.")

    # Aguardar interação com o usuário
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Exemplo de uso
img_path = 'cartao1.jpg'
identificar_alternativas(img_path)
