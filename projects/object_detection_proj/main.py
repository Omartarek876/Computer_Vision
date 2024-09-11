import cv2 as cv

"""
Object Detection Using OpenCV DNN and Webcam

Author: Omar Tarek Ibrahim
Description: - This project implements real-time object detection using a pre-trained object detection model (SSD MobileNet V3) and a webcam feed ,It captures video from a mobile camera via an IP stream, detects objects, and displays the results in a window.
             - The model is based on the COCO dataset and is capable of detecting and labeling objects with bounding boxes.
Last Updated: September 11, 2024

Requirements:
- Python 3.x
- OpenCV (cv2) library
- Pre-trained SSD MobileNet V3 model files
- `coco.names` file containing class names

Setup:
1. Install OpenCV: `pip install opencv-python`
2. Download the pre-trained model files and `coco.names` from the respective sources.
3. Update the IP URL in the `main` function with the IP address of your mobile camera (IP Webcam APP).

"""

def load_class_names(classfile):
    """
    Load class names from a file.

    Args:
    classfile (str): Path to the file containing class names, one per line.

    Returns:
    list: List of class names.
    """
    classnames = []
    with open(classfile, "rt") as f: 
        classnames = f.read().strip("\n").split("\n")
    return classnames

def initialize_model(weightsPth, configPath):
    """
    Initialize the object detection model using pre-trained weights and configuration files.

    Args:
    weightsPth (str): Path to the weights file, which contains the trained model parameters.
    configPath (str): Path to the configuration file that defines the model architecture.

    Returns:
    cv.dnn_DetectionModel: Initialized detection model object.
    """
    net = cv.dnn_DetectionModel(weightsPth, configPath)
    # Set input size of the network; this must match the input size the network was trained with
    net.setInputSize(320, 230)
    # Set scaling factor for input images; normalization used during training
    net.setInputScale(1.0 / 127.5)
    # Set mean subtraction values for input images
    net.setInputMean((127.5, 127.5, 127.5))
    # Swap Red and Blue channels if required by the network
    net.setInputSwapRB(True)
    return net

def detect_and_display_objects(cam, net, classnames):
    """
    Capture video frames from the camera, detect objects in each frame, and display the results.

    Args:
    cam (cv.VideoCapture): Video capture object for accessing the camera feed.
    net (cv.dnn_DetectionModel): Object detection model to process frames.
    classnames (list): List of class names corresponding to the object classes detected by the model.
    """
    i = 0
    while True:
        success, img = cam.read()  # Capture a frame from the camera
        if not success or img is None:
            print("Failed to capture image. Retrying...")
            continue  # Skip to the next frame if capture failed

        # Resize the image to a smaller size for display
        img_resized = cv.resize(img, (640, 480))  # Width x Height of the resized image

        # Perform object detection on the resized image
        classIds, confs, bbox = net.detect(img_resized, confThreshold=0.5)
        if len(classIds) != 0:  # Check if any objects were detected
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                x, y, w, h = box  # Unpack bounding box coordinates
                # Draw a rectangle around the detected object
                cv.rectangle(img_resized, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
                # Annotate the detected object with its class name
                cv.putText(img_resized, classnames[classId - 1], (x + 10, y + 20),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        # Display the frame with detection results
        cv.imshow("Object Detection", img_resized)
        
        key = cv.waitKey(1)  # Wait for a key press
        if key == ord('c'):
            # Save the current frame as an image when 'c' is pressed
            cv.imwrite(f"captured_images/captured_image{i}.jpg", img_resized)
            print("Image captured successfully!")
            i += 1  # Increment image counter
        if key == ord('q'):
            # Exit the loop when 'q' is pressed
            break

def main():
    """
    Main function to set up the camera, load the model, and start object detection.
    """
    # Replace the IP URL with the URL of your mobile camera's video stream
    ip_url = "http://192.168.1.8:8080/video"  # Example IP address
    cam = cv.VideoCapture(ip_url)  # Initialize video capture from the mobile camera

    # Load class names and initialize the detection model
    classnames = load_class_names("coco.names")
    net = initialize_model("frozen_inference_graph.pb", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    detect_and_display_objects(cam, net, classnames)

    # Release camera and close all OpenCV windows
    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
