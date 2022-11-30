from pathlib import Path
import cv2
import face_recognition as fr

# Cargar imagenes
foto_control = fr.load_image_file(Path('fotos', 'foto1.jpg'))
foto_prueba = fr.load_image_file(Path('fotos', 'foto2.jpg'))

foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

# Localizar cara de control
lugar_cara_1 = fr.face_locations(foto_control)[0]
lugar_cara_2 = fr.face_locations(foto_prueba)[0]
cara_codificada_1 = fr.face_encodings(foto_control)[0]

print(lugar_cara_1[3], lugar_cara_1[0])
# Mostrar rectángulo
cv2.rectangle(foto_control,
              (lugar_cara_1[3], lugar_cara_1[0]),
              (lugar_cara_1[1], lugar_cara_1[2]),
              (0, 255, 0), 2)

cv2.rectangle(foto_prueba,
              (lugar_cara_2[3], lugar_cara_2[0]),
              (lugar_cara_2[1], lugar_cara_2[2]),
              (0, 255, 0), 2)

# Mostrar imágenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)

# Mantener el programa abierto
cv2.waitKey(0)
