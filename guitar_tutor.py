import cv2
import numpy as np
import librosa
import librosa.display
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from watershed import watershed_segmentation
from collections import defaultdict
import numpy as np
import cv2
import librosa
import librosa.display
from sklearn.cluster import KMeans
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

def skin_detection(img):
    if img is None:
        print("Error: Input image is None.")
        return None
    for index_line, line in enumerate(img):
        for index_pixel, pixel in enumerate(line):
            if pixel[2] > 95 and pixel[1] > 40 and pixel[0] > 20 and max(pixel) - min(pixel) > 15 \
                    and abs(pixel[2] - pixel[1]) > 15 and pixel[2] > pixel[0] and pixel[2] > pixel[1] \
                    and index_pixel > len(line) / 2:
                pass
            else:
                img[index_line][index_pixel] = (0, 0, 0)
    return img

def locate_hand_region(img):
    if img is None:
        print("Error: Image is None in locate_hand_region().")
        return None, None
    height = len(img)
    width = len(img[0])
    hand_region = np.zeros((height, width, 3), np.uint8)

    x_dict = defaultdict(int)
    for line in img:
        for j, pixel in enumerate(line):
            if pixel.all() > 0:
                x_dict[j] += 1

    max_density = max(x_dict.values())
    max_x_density = 0
    for x, density in x_dict.items():
        if density == max_density:
            max_x_density = x
            break
    min_x = min(x_dict.keys())
    max_x = max(x_dict.keys())

    m = 0
    last_density = x_dict[max_density]
    while 1:
        if max_x_density - m == min_x:
            break
        m += 1
        current_density = x_dict[max_x_density - m]
        if current_density < 0.1 * max_density:
            break
        elif current_density < 0.5 * last_density:
            break
        last_density = current_density

    n = 0
    last_density = x_dict[max_density]
    while 1:
        if max_x_density + n == max_x:
            break
        n += 1
        current_density = x_dict[max_x_density + n]
        if current_density < 0.1 * max_density:
            break
        elif current_density < 0.5 * last_density:
            break
        last_density = current_density

    tolerance = 20
    min_limit = max_x_density - m - tolerance
    max_limit = max_x_density + n + tolerance

    for i, line in enumerate(img):
        for j, pixel in enumerate(line):
            if min_limit < j < max_limit:
                hand_region[i][j] = img[i][j]

    return hand_region

def extract_chords(audio_path):
    y, sr = librosa.load(audio_path)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    
    # KMeans clustering to identify chords
    kmeans = KMeans(n_clusters=24).fit(beat_chroma.T)
    chords = kmeans.labels_
    return chords, librosa.frames_to_time(beat_frames, sr=sr)


def detect_fretboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    # Assuming the fretboard lines are the most prominent
    return lines

def normalize_fretboard(image, lines):
    """
    Simplified placeholder for normalizing the fretboard.
    :param image: an image of the cropped neck
    :param lines: detected fretboard lines
    :return: normalized image (currently returns the input image)
    """
    try:
        print("normalize_fretboard called.")
        height, width = image.shape[:2]
        print(f"Image dimensions: height={height}, width={width}")
        
        if len(lines) < 2:
            print("Not enough lines detected to normalize fretboard.")
            return image  # Return the input image if not enough lines
        
        print("Detected lines for normalization:", lines)

        # Simply return the input image for now
        return image
    except Exception as e:
        print("Error in normalize_fretboard:", e)
        return image  # Return the input image in case of error

    """
    Normalize the fretboard based on detected lines.
    :param image: an image of the cropped neck
    :param lines: detected fretboard lines
    :return: normalized image
    """
    try:
        height, width = image.shape[:2]
        print(f"Image dimensions: height={height}, width={width}")
        
        if len(lines) < 2:
            print("Not enough lines detected to normalize fretboard.")
            return None
        
        print("Detected lines for normalization:", lines)

        # Example transformation matrix calculation (placeholder)
        # Note: This should be replaced with actual logic
        src_points = np.float32([lines[0][0], lines[1][0], lines[0][1], lines[1][1]])
        dst_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

        print("Source points for perspective transform:", src_points)
        print("Destination points for perspective transform:", dst_points)

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        print("Perspective transformation matrix:", matrix)

        normalized = cv2.warpPerspective(image, matrix, (width, height))
        print("Normalization successful.")
        return normalized
    except Exception as e:
        print("Error in normalize_fretboard:", e)
        return None


def detect_strings_and_frets(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    return lines

def detect_fingers(image):
    # Apply skin detection and locate the hand region
    skin = skin_detection(image)
    if skin is None:
        print("Skin detection failed.")
        return None
    hand_region = locate_hand_region(skin)
    if hand_region is None:
        print("Hand region location failed.")
        return None
    try:
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny edge detector
        edges = cv2.Canny(blur, 50, 150)
        
        # Detect circles using Hough transform
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=5, maxRadius=15)
        if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw the outer circle
                    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw the center of the circle
                    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        else:
            print("No circles detected.")

        return circles
    except Exception as e:
        print("Error in detect_fingers:", e)
        return None


def map_fingers_to_fretboard(fretboard, fingers):
    """
    Map finger positions to the detected fretboard grid.
    
    :param fretboard: The detected fretboard grid lines.
    :param fingers: The detected finger positions.
    :return: A list of mapped finger positions.
    """
    mapped_positions = []
    
    for finger in fingers:
        x, y = finger
        closest_fret = None
        min_dist = float('inf')
        
        # Find the closest fret line to the finger position
        for fret in fretboard:
            fret_x1, fret_y1, fret_x2, fret_y2 = fret
            dist = abs((fret_y2 - fret_y1) * x - (fret_x2 - fret_x1) * y + fret_x2 * fret_y1 - fret_y2 * fret_x1) / \
                   ((fret_y2 - fret_y1) ** 2 + (fret_x2 - fret_x1) ** 2) ** 0.5
            
            if dist < min_dist:
                min_dist = dist
                closest_fret = fret
        
        if closest_fret is not None:
            mapped_positions.append((x, closest_fret))
    
    return mapped_positions


def recognize_chord(fingers):
    """
    Recognize the chord based on finger positions.
    
    :param fingers: The detected finger positions.
    :return: The recognized chord name.
    """
    # Example chord mappings (fret position: (string, fret))
    chord_mappings = {
        'C': [(1, 0), (2, 1), (3, 0), (4, 2), (5, 3), (6, 0)],
        'G': [(1, 3), (2, 0), (3, 0), (4, 0), (5, 2), (6, 3)],
        'Am': [(1, 0), (2, 1), (3, 2), (4, 2), (5, 0), (6, 0)],
        'F': [(1, 1), (2, 1), (3, 2), (4, 3), (5, 3), (6, 1)]
    }
    
    # Convert finger positions to a set for comparison
    finger_set = set(fingers)
    
    for chord, positions in chord_mappings.items():
        if finger_set == set(positions):
            return chord
    
    return "Unknown"


def overlay_finger_positions(frame, finger_positions, expected_positions):
    for pos in expected_positions:
        cv2.circle(frame, pos, 10, (0, 0, 255), -1)
    for pos in finger_positions:
        if pos in expected_positions:
            cv2.circle(frame, pos, 10, (0, 255, 0), -1)
    return frame

from strings import Strings
from rotate_crop import *
import cv2
import numpy as np
from collections import defaultdict
from image import Image

def string_detection(neck):
    """
    Detecting and separating strings into separate blocks by choosing numerous vertical slices in image
    We then look for a periodical pattern on each slice (ie strings detected), store points in between strings
    and connect them using a regression fitting function to separe each string
    :param neck: An Image object of the picture cropped around the horizontal neck
    :return strings, Image_string: either a string object which is a dict associating each string to a line
    (ie a tuple of points) delimiting the bottom of the string block // or directly an Image object with those
    lines displayed (illustration purpose)
    """
    if neck.image is None:
        print("Error: neck image is None.")
        return None, None
    height = len(neck.image)
    width = len(neck.image[0])
    neck_with_strings = np.zeros((height, width, 3), np.uint8)

    # 1. Detect strings with Hough transform and form an Image based on these
    edges = neck.edges_sobely()
    edges = threshold(edges, 127)

    lines = neck.lines_hough_transform(edges, 50, 20)  # TODO: Calibrate params automatically if possible
    size = len(lines)

    for x in range(size):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(neck_with_strings, (x1, y1), (x2, y2), (255, 255, 255), 2)

    neck_str = Image(img=neck_with_strings)
    neck_str_gray = neck_str.gray

    # 2. Slice image vertically at different points and calculate gaps between strings at these slices
    slices = {}
    nb_slices = int(width / 50)
    for i in range(nb_slices):
        slices[(i + 1) * nb_slices] = []  # slices dict is {x_pixel_of_slice : [y_pixels_where_line_detected]}

    for index_line, line in enumerate(neck_str_gray):
        for index_pixel, pixel in enumerate(line):
            if pixel == 255 and index_pixel in slices:
                slices[index_pixel].append(index_line)

    slices_differences = {}  # slices_differences dict is {x_pixel_of_slice : [gaps_between_detected_lines]}
    for k in slices.keys():
        temp = []
        n = 0
        slices[k] = list(sorted(slices[k]))
        for p in range(len(slices[k]) - 1):
            temp.append(slices[k][p + 1] - slices[k][p])
            if slices[k][p + 1] - slices[k][p] > 1:
                n += 1
        slices_differences[k] = temp

    points = []
    points_dict = {}
    for j in slices_differences.keys():
        gaps = [g for g in slices_differences[j] if g > 1]
        points_dict[j] = []

        if len(gaps) > 3:
            median_gap = median(gaps)
            for index, diff in enumerate(slices_differences[j]):
                if abs(diff - median_gap) < 4:
                    points_dict[j].append((j, slices[j][index] + int(median_gap / 2)))
                elif abs(diff / 2 - median_gap) < 4:
                    points_dict[j].append((j, slices[j][index] + int(median_gap / 2)))
                    points_dict[j].append((j, slices[j][index] + int(3 * median_gap / 2)))

        points.extend(points_dict[j])

    '''for p in points:
        print(p)
        cv2.circle(neck.image, p, 3, (0, 255, 0), -1)
    plt.imshow(cv2.cvtColor(neck.image, cv2.COLOR_BGR2RGB))
    plt.show()'''

    points_divided = [[] for i in range(5)]
    for s in points_dict.keys():
        for i in range(5):
            try:
                # cv2.circle(neck.image, points_dict[s][i], 3, (255, 0, 0), -1)
                points_divided[i].append(points_dict[s][i])
            except IndexError:
                pass

    # 3. Use fitLine function to form lines separating each string

    tuning = ["E", "A", "D", "G", "B", "E6"]
    strings = Strings(tuning)

    for i in range(5):
        cnt = np.array(points_divided[i])
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L12, 0, 0.01, 0.01)  # best distType found was DIST_L12

        left_extreme = int((-x * vy / vx) + y)
        right_extreme = int(((width - x) * vy / vx) + y)

        strings.separating_lines[tuning[i]] = [(width - 1, right_extreme), (0, left_extreme)]

        cv2.line(neck.image, (width - 1, right_extreme), (0, left_extreme), (0, 0, 255), 2)

    return strings, Image(img=neck.image)

def fret_detection(image):
    """
    Simplified placeholder for detecting frets.
    :param image: an image of the normalized fretboard
    :return: image with detected frets (currently returns the input image)
    """
    try:
        print("fret_detection called.")
        height, width = image.shape[:2]
        print(f"Image dimensions: height={height}, width={width}")

        # Placeholder logic for fret detection
        # Assuming lines are detected and drawn directly for simplicity
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            print("No lines detected.")

        return image
    except Exception as e:
        print("Error in fret_detection:", e)
        return image  # Return the input image in case of error



if __name__ == "__main__":
    print("Run grid_detection_tests.py to have a look at results!")

def get_expected_positions(chord):
    """
    Get the expected finger positions for a given chord.
    :param chord: The chord for which to get finger positions.
    :return: A list of tuples representing the expected finger positions (fret, string).
    """
    chord_positions = {
        'C': [(1, 2), (2, 4), (3, 5)],  # (fret, string)
        'G': [(2, 5), (3, 6), (3, 1)],
        'D': [(2, 3), (3, 2), (2, 1)],
        'Am': [(1, 2), (2, 4), (2, 3)],
        # Add more chords as needed
    }

    return chord_positions.get(chord, [])


#def main(audio_path):
    '''
    def open_camera():
    # Open the default camera (usually the first camera found)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    '''
    chords, timings = extract_chords(audio_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    chord_index = 0
    current_chord = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame. Skipping...")
            continue
        
        image = Image(img=frame)
        if image.image is None:
            print("Frame capture failed.")
            continue

        rotated_image = rotate_neck_picture(image)
        if rotated_image.image is None:
            print("Failed to rotate image.")
            continue

        cropped_image = crop_neck_picture(rotated_image)
        if cropped_image is None or cropped_image.image is None:
            print("Failed to crop neck. Skipping frame.")
            continue
        
        lines = detect_fretboard(cropped_image.image)
        if lines is None:
            print("Failed to detect fretboard.")
            continue

        normalized_frame = normalize_fretboard(cropped_image.image, lines)
        if normalized_frame is None:
            print("Failed to normalize fretboard.")
            continue

        string_lines, string_image = string_detection(Image(img=normalized_frame))
        if string_lines is None or string_image is None:
            print("Failed to detect strings. Skipping frame.")
            continue

        fret_lines = fret_detection(Image(img=normalized_frame))
        if fret_lines is None:
            print("Failed to detect frets.")
            continue

        circles = detect_fingers(normalized_frame)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            finger_positions = [(i[0], i[1]) for i in circles[0, :]]
        else:
            finger_positions = []

        # Check if it's time to update the current chord
        if chord_index < len(timings) and timings[chord_index] <= cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:
            current_chord = chords[chord_index]
            chord_index += 1

        if current_chord is not None:
            expected_positions = get_expected_positions(current_chord)
            overlay_frame = overlay_finger_positions(normalized_frame, finger_positions, expected_positions)
        
            # Optional: Apply watershed segmentation for hand detection
            watershed_frame = watershed_segmentation(normalized_frame)
        
            cv2.imshow('Guitar Tutor', overlay_frame)
            cv2.imshow('Watershed Segmentation', watershed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


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

        # Convert frame to Image object
        image = Image(img=frame)
        if image.image is None:
            print("Frame capture failed.")
            continue
        
        # Rotate neck picture
        rotated_image = rotate_neck_picture(image)
        if rotated_image.image is None:
            print("Failed to rotate image.")
            continue

        # Display the rotated frame for debugging purposes
        cv2.imshow('Rotated Frame', rotated_image.image)

        # Crop neck picture
        cropped_image = crop_neck_picture(rotated_image)
        if cropped_image is None or cropped_image.image is None:
            print("Failed to crop neck. Skipping frame.")
            continue

        # Display the cropped frame for debugging purposes
        cv2.imshow('Cropped Frame', cropped_image.image)

        # Detect fretboard
        lines = detect_fretboard(cropped_image.image)
        if lines is None:
            print("Failed to detect fretboard.")
            continue

        # Normalize fretboard
        normalized_frame = normalize_fretboard(cropped_image.image, lines)
        if normalized_frame is None:
            print("Failed to normalize fretboard.")
            continue

        # Display the normalized frame for debugging purposes
        cv2.imshow('Normalized Frame', normalized_frame)

        # Detect strings
        string_lines, string_image = string_detection(Image(img=normalized_frame))
        if string_lines is None or string_image is None:
            print("Failed to detect strings. Skipping frame.")
            continue

        # Display the string detection frame for debugging purposes
        cv2.imshow('String Detection', string_image.image)

        # Detect frets
        fret_lines = fret_detection(Image(img=normalized_frame))
        if fret_lines is None:
            print("Failed to detect frets.")
            continue

        # Detect fingers
        circles = detect_fingers(normalized_frame)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            finger_positions = [(i[0], i[1]) for i in circles[0, :]]
        else:
            finger_positions = []

        # Check if it's time to update the current chord
        if chord_index < len(timings) and timings[chord_index] <= cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:
            current_chord = chords[chord_index]
            chord_index += 1

        if current_chord is not None:
            expected_positions = get_expected_positions(current_chord)
            overlay_frame = overlay_finger_positions(normalized_frame, finger_positions, expected_positions)
        
            # Optional: Apply watershed segmentation for hand detection
            watershed_frame = watershed_segmentation(normalized_frame)
        
            cv2.imshow('Guitar Tutor', overlay_frame)
            cv2.imshow('Watershed Segmentation', watershed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


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

        # Convert frame to Image object
        image = Image(img=frame)
        if image.image is None:
            print("Frame capture failed.")
            continue

        # Rotate neck picture
        rotated_image = rotate_neck_picture(image)
        if rotated_image.image is None:
            print("Failed to rotate image.")
            continue

        # Display the rotated frame for debugging purposes
        cv2.imshow('Rotated Frame', rotated_image.image)

        # Crop neck picture
        cropped_image = crop_neck_picture(rotated_image)
        if cropped_image is None or cropped_image.image is None:
            print("Failed to crop neck. Skipping frame.")
            continue

        # Display the cropped frame for debugging purposes
        cv2.imshow('Cropped Frame', cropped_image.image)

        # Detect fretboard
        lines = detect_fretboard(cropped_image.image)
        if lines is None:
            print("Failed to detect fretboard.")
            continue

        # Normalize fretboard
        normalized_frame = normalize_fretboard(cropped_image.image, lines)
        if normalized_frame is None:
            print("Failed to normalize fretboard.")
            continue

        # Display the normalized frame for debugging purposes
        cv2.imshow('Normalized Frame', normalized_frame)

        # Detect strings
        string_lines, string_image = string_detection(Image(img=normalized_frame))
        if string_lines is None or string_image is None:
            print("Failed to detect strings. Skipping frame.")
            continue

        # Display the string detection frame for debugging purposes
        cv2.imshow('String Detection', string_image.image)

        # Detect frets
        fret_lines = fret_detection(Image(img=normalized_frame))
        if fret_lines is None:
            print("Failed to detect frets.")
            continue

        # Display the fret detection frame for debugging purposes
        cv2.imshow('Fret Detection', normalized_frame)

        # Detect fingers
        circles = detect_fingers(normalized_frame)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            finger_positions = [(i[0], i[1]) for i in circles[0, :]]
        else:
            finger_positions = []

        # Display the finger detection frame for debugging purposes
        cv2.imshow('Finger Detection', normalized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main(audio_path):
    chords, timings = extract_chords(audio_path)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    chord_index = 0
    current_chord = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame. Skipping...")
            continue

        # Display the raw frame for debugging purposes
        cv2.imshow('Raw Frame', frame)

        # Convert frame to Image object
        image = Image(img=frame)
        if image.image is None:
            print("Frame capture failed.")
            continue

        # Rotate neck picture
        rotated_image = rotate_neck_picture(image)
        if rotated_image.image is None:
            print("Failed to rotate image.")
            continue

        # Display the rotated frame for debugging purposes
        cv2.imshow('Rotated Frame', rotated_image.image)

        # Crop neck picture
        cropped_image = crop_neck_picture(rotated_image)
        if cropped_image is None or cropped_image.image is None:
            print("Failed to crop neck. Skipping frame.")
            continue

        # Display the cropped frame for debugging purposes
        cv2.imshow('Cropped Frame', cropped_image.image)

        # Detect fretboard
        lines = detect_fretboard(cropped_image.image)
        if lines is None:
            print("Failed to detect fretboard.")
            continue

        print("Fretboard detected. Lines:", lines)

        # Normalize fretboard
        normalized_frame = normalize_fretboard(cropped_image.image, lines)
        if normalized_frame is None:
            print("Failed to normalize fretboard.")
            continue

        # Display the normalized frame for debugging purposes
        cv2.imshow('Normalized Frame', normalized_frame)

        # Detect strings
        try:
            string_lines, string_image = string_detection(Image(img=normalized_frame))
            if string_lines is None or string_image is None:
                print("Failed to detect strings. Skipping frame.")
                continue

            # Display the string detection frame for debugging purposes
            cv2.imshow('String Detection', string_image.image)
        except Exception as e:
            print("Exception occurred while detecting strings:", e)
            continue

        # Detect frets
        try:
            fret_image = fret_detection(normalized_frame)
            if fret_image is None:
                print("Failed to detect frets.")
                continue

            # Display the fret detection frame for debugging purposes
            cv2.imshow('Fret Detection', fret_image)
        except Exception as e:
            print("Exception occurred while detecting frets:", e)
            continue

        # Detect fingers
        try:
            circles = detect_fingers(normalized_frame)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                finger_positions = [(i[0], i[1]) for i in circles[0, :]]
            else:
                finger_positions = []

            # Display the finger detection frame for debugging purposes
            cv2.imshow('Finger Detection', normalized_frame)
        except Exception as e:
            print("Exception occurred while detecting fingers:", e)
            continue

        # Check if it's time to update the current chord
        if chord_index < len(timings) and timings[chord_index] <= cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:
            current_chord = chords[chord_index]
            chord_index += 1

        if current_chord is not None:
            expected_positions = get_expected_positions(current_chord)
            overlay_frame = overlay_finger_positions(normalized_frame, finger_positions, expected_positions)
        
            # Optional: Apply watershed segmentation for hand detection
            watershed_frame = watershed_segmentation(normalized_frame)
        
            cv2.imshow('Guitar Tutor', overlay_frame)
            cv2.imshow('Watershed Segmentation', watershed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows() 




import numpy as np
if __name__ == "__main__":
    audio_path = '/Users/chethana/Downloads/Coldplay - Yellow (Official Video).mp3'
    main(audio_path)