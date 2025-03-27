import cv2

# ----------- READ DNN MODEL ----------- 
prototxt = "modelo/MobileNetSSD_deploy.prototxt.txt"  # Asegúrate de que el archivo esté en la ubicación correcta
model = "modelo/MobileNetSSD_deploy.caffemodel"  # Asegúrate de que el archivo esté en la ubicación correcta

# Clases de objetos que el modelo puede detectar
classes = {0: "background", 1: "aeroplane", 2: "bicycle",
           3: "bird", 4: "boat", 5: "bottle", 6: "bus", 
           7: "car", 8: "cat", 9: "chair", 10: "cow", 
           11: "diningtable", 12: "dog", 13: "horse", 
           14: "motorbike", 15: "person", 16: "pottedplant", 
           17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}

# Cargar el modelo DNN
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# ----------- READ THE IMAGE AND PREPROCESSING ----------- 
# Si es un archivo de video, usa cv2.VideoCapture
cap = cv2.VideoCapture("imagenesVideos/h3a3770fc_A094_02131509_C141.mov")  # Cambia el nombre del archivo si es necesario

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    image_resized = cv2.resize(frame, (300, 300))

    # Crear un blob
    blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    
    # Configurar la red con el blob
    net.setInput(blob)
    
    # Obtener las detecciones
    detections = net.forward()

    # ----------- DETECTIONS AND PREDICTIONS ----------- 
    for i in range(detections.shape[2]):
        detection = detections[0, 0, i]
        confidence = detection[2]
        
        if confidence > 0.45:  # Umbral de confianza
            label = classes[int(detection[1])]
            print("Label:", label)
            
            # Coordenadas de la caja delimitadora
            box = detection[3:7] * [width, height, width, height]
            x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Dibujar la caja y el texto
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence * 100:.2f}%", (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, label, (x_start, y_start - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Mostrar la imagen o el video procesado
    cv2.imshow("Detected Objects", frame)
    
    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
