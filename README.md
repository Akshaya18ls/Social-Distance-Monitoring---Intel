# **Social Distance Monitoring System**

This project implements a real-time social distancing monitoring system using object detection, distance calculation, and alert mechanisms. It uses YOLO (You Only Look Once) for detecting people in video frames 
and calculates the distance between detected individuals to ensure compliance with social distancing guidelines. If the distance between any two people falls below a defined threshold, an alert is triggered and 
sent to an Arduino to trigger a visual or audible notification.

## **Features**

•	Detects individuals in video frames using YOLOv3.

•	Measures the distance between detected individuals.

•	Triggers an alert if the distance between any two people falls below the threshold.

•	Sends an alert signal to an Arduino or ESP32 device via serial communication to activate a buzzer or LED.

•	Displays clusters of people using KMeans clustering.

## **Technologies Used**

•	Python: Primary programming language.

•	OpenCV: For video processing and computer vision tasks.

•	YOLOv3: Object detection model.

•	KMeans Clustering: For visualizing clusters of people.

•	Serial Communication: Sends alerts via serial communication to external devices like Arduino.

•	Sklearnex: For optimized machine learning performance.

•	Imutils: For resizing frames and image processing utilities.

## **Installation**

1.	Clone the repository:
```bash
git clone https://github.com/yourusername/social-distance-monitoring.git
```

2.	Install dependencies:
```bash
pip install -r requirements.txt
```
Sample requirements.txt:
```txt
opencv-python
numpy
imutils
pyserial
scikit-learn
scikit-learn-intelex
```

3.	Download the YOLOv3 weights and configuration files: yolov3.weights, yolov3.cfg

4.	Place the YOLO files (yolov3.weights and yolov3.cfg) in the same directory as the project script.

5.	Connect your Arduino or ESP32 to the appropriate COM port (e.g., COM7).

## **Connecting the Arduino**

This project includes sending an alert signal to an Arduino via serial communication. The Arduino can be connected to trigger a buzzer or LED when social distancing violations are detected.
### **Components**

•	Arduino Uno or ESP32

•	Buzzer or LED for visual/audible alert

•	USB cable to connect Arduino to your computer

### **Steps**

1.	Arduino Wiring:
o	Buzzer: Connect the positive leg of the buzzer to digital pin 13, and the negative leg to GND.
o	LED: Connect the longer leg (positive) of the LED to pin 13 via a 220-ohm resistor, and the shorter leg (negative) to GND.

2.	Arduino Code: Upload the following code to your Arduino:
```cpp
int alertPin = 13;

void setup() {
  Serial.begin(9600); 
  pinMode(alertPin, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char receivedData = Serial.read();

    if (receivedData == '1') {
      digitalWrite(alertPin, HIGH);  // Trigger the buzzer/LED
    } else if (receivedData == '0') {
      digitalWrite(alertPin, LOW);   // Turn off the buzzer/LED
    }
  }
}
```

3.	Set Up Serial Communication in the Python script:
```python
ser = serial.Serial('COM7', 9600, timeout=1)
```

## **Usage**

1.	Run the Python script:
```bash
python social_distance_monitor.py
```

2.	The script will:
o	Open a video file (test.mp4) and start monitoring the people in the video for social distancing.
o	Detect people, calculate the distance between them, and send a signal ('1' or '0') to the Arduino to trigger an alert if necessary.

3.	Press q to exit the program.

## **Parameters**

•	DISTANCE_THRESHOLD: Minimum distance (in pixels) between people for the system to trigger an alert. Default is 25 pixels.

•	MAX_FRAMES: Maximum number of frames to process before exiting.

## **Customization**

•	Change Video Input: Update the following line with the path to your video:
```python
cap = cv2.VideoCapture(r"path_to_your_video")
```

•	Modify Distance Threshold: Change the DISTANCE_THRESHOLD value to customize the alert trigger:
```python
DISTANCE_THRESHOLD = 25
```

## **Dependencies**

•	OpenCV: For video processing and object detection.

•	NumPy: For numerical operations.

•	Imutils: For resizing video frames.

•	PySerial: For serial communication with Arduino/ESP32.

•	Scikit-Learn: For KMeans clustering.

