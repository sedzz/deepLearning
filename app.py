import cv2
import dlib
import time
import winsound
import numpy as np
from scipy.spatial import distance as dist
from datetime import datetime

# Cargar modelo de de puntos faciales del ojo
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./src/shape_model.dat')

# Proporcion del ojo
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Índices para los puntos faciales correspondientes a los ojos
(lStart, lEnd) = (36, 42)
(rStart, rEnd) = (42, 48)

# Umbral deteccion cuando el ojo esta cerrado
EAR_THRESHOLD = 0.27
# Altura del ojo (? fase pruebas)
EYE_HEIGHT_THRESHOLD = 5.0  # Ajusta este valor según sea necesario
# Numero de FPS para determinar si el conductor esta dormido
EAR_CONSEC_FRAMES = 5

# Inicializar contadores y estado
frame_counter = 0
start_time = None
eyes_closed = False
beeping = False

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error al capturar el frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("No se detectaron rostros")

    for face in faces:
        print("Rostro detectado")
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        left_eye = landmarks[lStart:lEnd]
        right_eye = landmarks[rStart:rEnd]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        ear = (left_ear + right_ear) / 2.0
        
        # Calcular la altura del ojo
        left_eye_height = dist.euclidean(left_eye[1], left_eye[5])
        right_eye_height = dist.euclidean(right_eye[1], right_eye[5])
        eye_height = (left_eye_height + right_eye_height) / 2.0
        
        print(f"EAR: {ear}, Altura del ojo: {eye_height}")

        # Dibujar los contornos de los ojos
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        if ear <= EAR_THRESHOLD or eye_height < EYE_HEIGHT_THRESHOLD:
            frame_counter += 1
            print(f"Contador de cuadros con ojos cerrados: {frame_counter}")

            if frame_counter >= EAR_CONSEC_FRAMES:
                if not beeping:
                    beeping = True
                    print("Ojos cerrados detectados: ", datetime.now())
                    winsound.Beep(1000, 1000)  # Beep at 1000 Hz for 1 second
                    print("----------------------------")
        else:
            frame_counter = 0
            beeping = False
            print("Ojos abiertos")

    cv2.imshow('Camara', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
