from pathlib import Path
import cv2
import face_recognition as fr

# Cargar imagenes
foto_control = fr.load_image_file(Path('fotos', 'foto1.jpg'))
foto_prueba = fr.load_image_file(Path('fotos', 'foto2.jpg'))

foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

# Mostrar imágenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)

# Mantener el programa abierto
cv2.waitKey(0)
