import cv2  # Importa a biblioteca OpenCV para processamento de imagem
import pickle  # Importa a biblioteca pickle para serialização/deserialização de objetos Python
import extrairGabarito as exG  # Importa um módulo externo chamado extrairGabarito para processamento específico

# Carrega os campos da imagem previamente processada
campos = []
with open('campos.pkl', 'rb') as arquivo:
    campos = pickle.load(arquivo)

# Carrega as respostas previamente processadas
resp = []
with open('resp.pkl', 'rb') as arquivo:
    resp = pickle.load(arquivo)

# Respostas corretas predefinidas para fins de comparação
respostasCorretas = ["1-B", "2-C", "3-A", "4-D", "5-B"]

# Inicializa a captura de vídeo usando a câmera indexada como 3 (pode ser ajustado conforme necessário)
video = cv2.VideoCapture(3)
