#When press s saved left camera and right camera frames to selected files

from picamera2 import Picamera2
import cv2
import os
import threading
import time

# Configuration settings
cameras_open = False  # Flag to control the camera loop
cameras_thread = None  # Thread for the camera loop
resolution = (1640, 1232)  # Default resolution
frame_rate = 25  # Target frame rate
display_size = (640, 480)  # Size for displaying the video feed

# Define the paths to save the images
left_img_path = '/home/lab902/Documents/Images/Cam0'
right_img_path = '/home/lab902/Documents/Images/Cam1'

# Create directories if they do not exist
os.makedirs(left_img_path, exist_ok=True)
os.makedirs(right_img_path, exist_ok=True)

# Initialize the Pi Cameras
cam0 = Picamera2(camera_num=0)
cam1 = Picamera2(camera_num=1)

# Configure the cameras with the specified resolution and frame rate
cam0.configure(cam0.create_still_configuration(main={"size": resolution, "format": "RGB888"}))
cam1.configure(cam1.create_still_configuration(main={"size": resolution, "format": "RGB888"}))

# Start the cameras
cam0.start()
cam1.start()

num = 0

def camera_loop():
    global cameras_open, num
    while cameras_open:
        # Capture frame-by-frame
        img0 = cam0.capture_array()
        img1 = cam1.capture_array()

        # Resize images for display
        img0_display = cv2.resize(img0, display_size)
        img1_display = cv2.resize(img1, display_size)

        # Display the images
        cv2.imshow('Left Camera', img0_display)
        cv2.imshow('Right Camera', img1_display)

        k = cv2.waitKey(5)

        if k == 27:  # Press 'Esc' key to exit
            cameras_open = False
            break
        elif k == ord('s'):  # Press 's' key to save the images
            cv2.imwrite(f'{left_img_path}/img{num}.png', img0)
            cv2.imwrite(f'{right_img_path}/img{num}.png', img1)
            print("Images saved!")
            num += 1

        time.sleep(1 / frame_rate)

# Start the camera loop in a separate thread
cameras_open = True
cameras_thread = threading.Thread(target=camera_loop)
cameras_thread.start()

# Wait for the camera loop to finish
cameras_thread.join()

# Release and destroy all windows before termination
cam0.stop()
cam1.stop()
cv2.destroyAllWindows()
