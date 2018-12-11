import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json


facePath = "haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(facePath)

model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5')  # carrega os pesos

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()  # Captura frame-by-frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(55, 55), flags=0)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        detected_face = frame[int(y):int(y + h), int(x):int(x + w)]  # detecta a face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transforma em escala de cinza
        detected_face = cv2.resize(detected_face, (48, 48))  # redimencionada a imagem

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # normaliza o valor

        predictions = model.predict(img_pixels)  # efetua a predicao

        # encontra o valor 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])

        emotion = emotions[max_index]

        # imprime o 'sentimento' acima da caixa
        cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Smile Detector', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
