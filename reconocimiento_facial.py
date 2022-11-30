from pathlib import Path
import cv2
import face_recognition as fr


# Cargar imagenes
def cargar_imagenes(path_list):
    # La primera será una foto de control, el resto de pruebas
    fotos = []
    for path in path_list:
        fotos.append(fr.load_image_file(path))
    return fotos


def asignar_perfil_color(fotos_list):
    for i in range(len(fotos_list)):
        fotos_list[i] = cv2.cvtColor(fotos_list[i], cv2.COLOR_BGR2RGB)
    return fotos_list


# top, right, botton, left
def localizar_cara(fotos_list):
    locations = []
    for i in fotos_list:
        locations.append(fr.face_locations(i)[0])
    return locations


def get_cod_faces(fotos_list):
    cod_faces = []
    for i in fotos_list:
        cod_faces.append(fr.face_encodings(i)[0])
    return cod_faces

# (left, top), (right, bottom)
def draw_rectangles(fotos_list, locations):
    for (f, l) in zip(fotos_list, locations):
        cv2.rectangle(f,
                      (l[3], l[0]),
                      (l[1], l[2]),
                      (0, 255, 0), 2)


def show_imgs(fotos_list):
    for index, f in enumerate(fotos_list):
        cv2.imshow(f'Foto {index}', f)


def compare_all_with_control(cara_cod_list):
    results = []
    for i,fc in enumerate(cara_cod_list):
        if i > 0:
            results.append(fr.compare_faces([cara_cod_list[0]], fc))
    return results



# foto_control = fr.load_image_file(Path('fotos', 'foto1.jpg'))
# foto_prueba = fr.load_image_file(Path('fotos', 'foto2.jpg'))

# foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
# foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

# Localizar cara de control
#lugar_cara_1 = fr.face_locations(foto_control)[0]
#lugar_cara_2 = fr.face_locations(foto_prueba)[0]
#cara_codificada_1 = fr.face_encodings(foto_control)[0]
#cara_codificada_2 = fr.face_encodings(foto_prueba)[0]

#print(lugar_cara_1[3], lugar_cara_1[0])
# Mostrar rectángulo
#cv2.rectangle(foto_control,
 #             (lugar_cara_1[3], lugar_cara_1[0]),
 #             (lugar_cara_1[1], lugar_cara_1[2]),
 #             (0, 255, 0), 2)

#cv2.rectangle(foto_prueba,
  #            (lugar_cara_2[3], lugar_cara_2[0]),
  #            (lugar_cara_2[1], lugar_cara_2[2]),
  #            (0, 255, 0), 2)

# Realizar comparación
#resultado = fr.compare_faces([cara_codificada_1], cara_codificada_2)
#print(resultado)

# Mostrar imágenes
# cv2.imshow('Foto Control', foto_control)
# cv2.imshow('Foto Prueba', foto_prueba)

# Mantener el programa abierto

