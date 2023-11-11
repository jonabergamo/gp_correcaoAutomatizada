import cv2
import numpy as np

def extrairMaiorCtn(img):
    """
    Extrai a região de interesse (ROI) correspondente ao maior contorno em uma imagem.

    Parâmetros:
    - img: Imagem de entrada (colorida).

    Retorno:
    - recorte: Imagem recortada contendo o maior contorno.
    - bbox: Lista contendo as coordenadas (x, y, largura, altura) da caixa delimitadora do maior contorno.
    """
    
    # Converte a imagem para tons de cinza
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplica threshold adaptativo para binarizar a imagem
    imgTh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 12)
    
    # Aplica dilatação para unir regiões próximas
    kernel = np.ones((2,2), np.uint8)
    imgDil = cv2.dilate(imgTh, kernel)
    
    # Encontra contornos na imagem dilatada
    contours, _ = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Encontra o maior contorno com base na área
    maiorCtn = max(contours, key=cv2.contourArea)
    
    # Obtém as coordenadas da caixa delimitadora do maior contorno
    x, y, w, h = cv2.boundingRect(maiorCtn)
    
    # Define a caixa delimitadora (bbox)
    bbox = [x, y, w, h]
    
    # Recorta a região de interesse (ROI) da imagem original
    recorte = img[y:y+h, x:x+w]
    
    # Redimensiona o recorte para as dimensões desejadas
    recorte = cv2.resize(recorte, (400, 500))

    # Retorna o recorte e a caixa delimitadora
    return recorte, bbox
