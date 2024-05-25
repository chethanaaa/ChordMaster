# import cv2

# def open_camera():
#     # Open the default camera (usually the first camera found)
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Error: Could not open video stream.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame. Exiting...")
#             break

#         # Display the resulting frame
#         cv2.imshow('Camera Feed', frame)

#         # Exit the loop when 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # When everything is done, release the capture
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     open_camera()
import cv2
import numpy as np
from image import Image
from functions import extract_chords, rotate_neck_picture, crop_neck_picture, detect_fretboard, normalize_fretboard, string_detection, fret_detection, detect_fingers, get_expected_positions, overlay_finger_positions, watershed_segmentation

def main(audio_path):
    chords, timings = extract_chords(audio_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame. Skipping...")
            continue

        # Display the raw frame for debugging purposes
        cv2.imshow('Raw Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    audio_path = '/Users/chethana/Downloads/Coldplay - Yellow (Official Video).mp3'
    main(audio_path)
