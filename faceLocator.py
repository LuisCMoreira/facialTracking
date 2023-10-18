import cv2
import dlib
import PySimpleGUI as sg
import numpy as np
import requests


def draw_crosshair(frame, x, y, size=20, color=(0, 255, 0), thickness=2):
    # Draw vertical line
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
    # Draw horizontal line
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)


def estimar_direcao_olhar(shape):
    boca = shape.part(48).x, shape.part(48).y, shape.part(54).x, shape.part(54).y
    nariz = shape.part(27).x, shape.part(27).y, shape.part(35).x, shape.part(35).y

    boca_centro_x = int((boca[0] + boca[2]) / 2)
    boca_centro_y = int((boca[1] + boca[3]) / 2)
    nariz_centro_x = int((nariz[0] + nariz[2]) / 2)
    nariz_centro_y = int((nariz[1] + nariz[3]) / 2)

    direcao_horizontal = boca_centro_x - nariz_centro_x
    
    msg1=""
    msg2=""

    if direcao_horizontal < -5:
        msg1= "Olhando para a esquerda"
    elif direcao_horizontal > 5:
        msg1= "Olhando para a direita"
    else:
        msg1="Olhando para a frente"

    
    direcao_vertical= boca_centro_y - nariz_centro_y

    if direcao_vertical < -5:
        msg2= "Olhando para a cima"
    elif direcao_vertical > 5:
        msg2= "Olhando para a baixo"
    else:
        msg2= "Olhando para a frente"
    
    msg=msg1 + " " + msg2 + " "  + str(boca_centro_x) + " " + str(boca_centro_y)
    return msg

def post_to_server(url, data):
    try:
        response = requests.post(url, json=data)

        # Check the status code to verify if the request was successful
        if response.status_code == 200:
            print("POST request was successful!")
        else:
            print(f"POST request failed with status code: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'.\shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

# Get the size (width and height) of the video frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video width: {width}, Video height: {height}")

layout = [
    [sg.Image(filename='', key='-IMAGE-')],
    [sg.Text('Direção do Olhar:', size=(15, 1)), sg.Text('', key='-DIRECTION-')]
]

window = sg.Window('Detecção de Direção do Olhar', layout, finalize=True)

while True:
    event, values = window.read(timeout=0, timeout_key='timeout')

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)

        direcao_olhar = estimar_direcao_olhar(shape)

        for i in range(68):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
            #print(x)
            #print(y)

        window['-DIRECTION-'].update(direcao_olhar)
        
        draw_crosshair(frame, x, y)
        
        # URL of the server where you want to post the data
        url = "http://127.0.0.1:81/api/jt1Control"

        # Data to be sent in the request (can be a dictionary or any JSON serializable object)
        data = {
            "key1": "66"
        }

        post_to_server(url, data)

    img_bytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=img_bytes)

    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break

cap.release()
cv2.destroyAllWindows()
window.close()