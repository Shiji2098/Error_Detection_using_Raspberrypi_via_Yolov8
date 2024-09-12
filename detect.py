import cv2

from ultralytics import YOLO



# Load your trained YOLOv8 model

model = YOLO('/home/laser/best.pt')



# Define GStreamer pipeline for Raspberry Pi camera

gst_pipeline = (

    "libcamerasrc ! video/x-raw,format=UYVY,width=640,height=480,framerate=30/1 ! videoconvert ! appsink"

)

# Open the Raspberry Pi camera using GStreamer pipeline

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)



if not cap.isOpened():

    print("Error: Could not open video stream")

    exit()



while True:

    # Capture frame-by-frame

    ret, frame = cap.read()

    if not ret:

        print("Error: Failed to grab frame")

        break



    # Run YOLOv8 inference on the frame

    results = model.predict(frame, imgsz=320, conf=0.25)

    result_frame = results[0].plot()



    # Display the results

    cv2.imshow('Real-time Detection', result_frame)



    # Press 'q' to exit

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



cap.release()

cv2.destroyAllWindows()

