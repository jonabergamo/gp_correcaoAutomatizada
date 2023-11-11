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
video = cv2.VideoCapture(0)

while True:
    # Captura frame do vídeo
    _, imagem = video.read()

    # Redimensiona a imagem
    imagem = cv2.resize(imagem, (600, 700))

    # Extrai gabarito e bounding box
    gabarito, bbox = exG.extrairMaiorCtn(imagem)

    # Converte gabarito para escala de cinza
    imgGray = cv2.cvtColor(gabarito, cv2.COLOR_BGR2GRAY)

    # Aplica limiarização binária inversa
    ret, imgTh = cv2.threshold(imgGray, 70, 255, cv2.THRESH_BINARY_INV)

    # Desenha retângulo ao redor do gabarito
    cv2.rectangle(imagem, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255,0), 3)

    # Inicializa lista de respostas
    respostas = []

    # Itera sobre os campos
    for id, vg in enumerate(campos):
        x = int(vg[0])
        y = int(vg[1])
        w = int(vg[2])
        h = int(vg[3])
        
        # Desenha retângulo ao redor do campo no gabarito
        cv2.rectangle(gabarito, (x, y), (x + w, y + h),(0,0,255),2)
        cv2.rectangle(imgTh, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # Extrai região de interesse (ROI)
        campo = imgTh[y:y + h, x:x + w]

        # Calcula a porcentagem de pixels pretos na região
        height, width = campo.shape[:2]
        tamanho = height * width
        pretos = cv2.countNonZero(campo)
        percentual = round((pretos / tamanho) * 100, 2)

        # Se a porcentagem de pixels pretos for suficiente, marca como resposta
        if percentual >=15:
            cv2.rectangle(gabarito, (x, y), (x + w, y + h), (255, 0, 0), 2)
            respostas.append(resp[id])

#print(respostas)
    erros = 0
    acertos = 0
    if len(respostas)==len(respostasCorretas):
        for num,res in enumerate(respostas):
            if res == respostasCorretas[num]:
                #print(f'{res} Verdadeiro, correto: {respostasCorretas[num]}')
                acertos +=1
            else:
                #print(f'{res} Falso, correto: {respostasCorretas[num]}')
                erros +=1

        pontuacao = int(acertos / len(respostasCorretas) * 10)
        cv2.putText(imagem,f'ACERTOS: {acertos}, PONTOS: {pontuacao}',(30,140),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

    cv2.imshow('img',imagem)
    cv2.imshow('Gabarito', gabarito)
    cv2.imshow('IMG TH', imgTh)
    cv2.waitKey(1)