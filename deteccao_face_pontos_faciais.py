import dlib
import cv2

def imprimePontos(imagem, pontosFaciais):
    for p in pontosFaciais.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 255), 2)


fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
imagem = cv2.imread("fotos/treinamento/ronald.0.0.jpg")
detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
facesDetectadas = detectorFace(imagem, 2)

# bounding box
for face in facesDetectadas:
    pontos = detectorPontos(imagem, face)
    print(pontos.parts())
    print(len(pontos.parts()))
    imprimePontos(imagem, pontos)

cv2.imshow("Pontos faciais", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
