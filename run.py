import cv2
import numpy as np
import imutils
import serial  
from sklearn.cluster import KMeans
from sklearnex import patch_sklearn

patch_sklearn()

def calculate_distance(point1, point2):
    return int(np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))

def main():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    ser = serial.Serial('COM7', 9600, timeout=1)  

    DISTANCE_THRESHOLD = 25
    cap = cv2.VideoCapture(r"C:\Users\amm20\OneDrive\Desktop\Social Distance Alerter\Clg.mp4")  

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        exit()

    frame_count = 0
    max_frames = 100  

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            frame_count += 1
            if frame_count > max_frames:
                print("Max frames reached. Exiting.")
                break

            frame = imutils.resize(frame, width=800)
            height, width, _ = frame.shape

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outputs = net.forward(output_layers)

            boxes = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and class_id == 0: 
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            if len(indexes) > 0:
                points = []
                alert_triggered = False
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    points.append((x + w // 2, y + h // 2))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        distance = calculate_distance(points[i], points[j])
                        if distance < DISTANCE_THRESHOLD:
                            cv2.line(frame, points[i], points[j], (0, 0, 255), 2)
                            alert_triggered = True

                if alert_triggered:
                    print("Alert triggered!")
                    ser.write(b'1')  
                    cv2.putText(frame, "Alert!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)  
                else:
                    ser.write(b'0')  

                if len(points) >= 2:  
                    kmeans = KMeans(n_clusters=2)  
                    clusters = kmeans.fit_predict(points)

                    for i, point in enumerate(points):
                        cv2.putText(frame, f"C{clusters[i]}", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Social Distancing Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        ser.close()  

if __name__ == "__main__":
    main()