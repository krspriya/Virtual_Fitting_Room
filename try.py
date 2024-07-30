import os
import cv2
from cvzone.PoseModule import PoseDetector
import pyautogui
import numpy as np
import mediapipe as mp
import time

# Get screen resolution
screenWidth, screenHeight = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screenWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screenHeight)

detector = PoseDetector()

# Path to shirt images
shirtFolderPath = "C:\\Users\\sssnj\\OneDrive\\Desktop\\3D Trail Room\\Resources\\Shirts"
listShirts = os.listdir(shirtFolderPath)

# Increase shirt size parameters
widthOfShirt = 400  # Increased width for the shirt
shirtRatioHeightWidth = 1.2 # Adjusted ratio to modify the height relative to the new width

# Initialize image number for shirt selection
imageNumber = 0

# Load overlay images and buttons
imgButtonRight = cv2.imread("C:\\Users\\sssnj\\OneDrive\\Desktop\\3D Trail Room\\Resources\\button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)

# Resize the overlay images
overlay_width = 128
overlay_height = 128
imgButtonRight = cv2.resize(imgButtonRight, (overlay_width, overlay_height))[:, :, :3]
imgButtonLeft = cv2.resize(imgButtonLeft, (overlay_width, overlay_height))[:, :, :3]

# Variables for button interaction
counterRight = 0
counterLeft = 0
selectionSpeed = 6  # Adjust as needed

# Load virtual background image
background_path = "C:\\Users\\sssnj\\OneDrive\\Desktop\\3D Trail Room\\Resources\\images (1).jpeg"
virtual_background = cv2.imread(background_path)
virtual_background = cv2.resize(virtual_background, (screenWidth, screenHeight))

# Initialize Mediapipe for segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# For smoother transitions
previous_mask = None

# Initialize cart state
cart = []
cart_icon_path = "C:\\Users\\sssnj\\OneDrive\\Desktop\\3D Trail Room\\Resources\\cart.jpg"
cart_icon = cv2.imread(cart_icon_path, cv2.IMREAD_UNCHANGED)
if cart_icon is None:
    raise FileNotFoundError(f"Cart icon not found at {cart_icon_path}")
cart_icon = cv2.resize(cart_icon, (50, 50))

# Notification state
notification_text = ""
notification_timer = 0

# Variables to manage pointing gesture
pointing_direction = None
pointing_threshold = 50  # Threshold for gesture detection

while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image from webcam.")
        continue

    # Resize virtual background to match the frame size
    img = cv2.resize(img, (screenWidth, screenHeight))

    # Perform segmentation
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(img_rgb)

    # Smooth the segmentation mask
    mask = results.segmentation_mask
    if previous_mask is not None:
        mask = cv2.addWeighted(mask, 0.6, previous_mask, 0.4, 0)
    previous_mask = mask
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    condition = np.stack((mask,) * 3, axis=-1) > 0.1
    output_image = np.where(condition, img, virtual_background)

    try:
        # Detect pose and landmarks
        output_image = detector.findPose(output_image)
        lmList, bboxInfo = detector.findPosition(output_image, bboxWithHands=False, draw=False)

        # Detect hands and landmarks
        hand_results = hands.process(img_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(output_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_landmarks = [[landmark.x * screenWidth, landmark.y * screenHeight] for landmark in hand_landmarks.landmark]

                # Determine pointing direction with a threshold
                index_finger_x = hand_landmarks[8][0]
                wrist_x = hand_landmarks[5][0]

                if index_finger_x - wrist_x > pointing_threshold:  # Right hand pointing right
                    pointing_direction = 'right'
                elif wrist_x - index_finger_x > pointing_threshold:  # Left hand pointing left
                    pointing_direction = 'left'
                else:
                    pointing_direction = None

                # Change shirt based on pointing direction
                if pointing_direction == 'right':
                    if imageNumber < len(listShirts) - 1:
                        imageNumber += 1
                    notification_text = f"Shirt changed to: {listShirts[imageNumber]}"
                    notification_timer = time.time()
                elif pointing_direction == 'left':
                    if imageNumber > 0:
                        imageNumber -= 1
                    notification_text = f"Shirt changed to: {listShirts[imageNumber]}"
                    notification_timer = time.time()

                # Check for other gestures
                fingers = [hand_landmarks[i][1] < hand_landmarks[i - 2][1] for i in [8, 12, 16, 20]]
                thumb = hand_landmarks[4][0] > hand_landmarks[3][0]
                if all(fingers) and thumb:
                    # Take a screenshot
                    screenshot_path = "screenshot.png"
                    cv2.imwrite(screenshot_path, output_image)
                    notification_text = "Screenshot taken!"
                    notification_timer = time.time()
                elif len(fingers) > 1 and all(fingers[:-1]) and not fingers[-1]:
                    # Wave gesture (simple version)
                    if hand_landmarks[8][0] > hand_landmarks[4][0] + 50:
                        # Add to cart
                        cart.append(listShirts[imageNumber])
                        notification_text = f"Added to cart: {listShirts[imageNumber]}"
                        notification_timer = time.time()

        if lmList:
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

            # Manually set shirt size
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))

            # Calculate centered position for the shirt
            shirt_height, shirt_width, _ = imgShirt.shape
            top_left_x = (screenWidth - shirt_width) // 2
            top_left_y = (screenHeight - shirt_height) // 2
            bottom_right_x = top_left_x + shirt_width
            bottom_right_y = top_left_y + shirt_height

            # Ensure coordinates are within the image boundaries
            top_left_x = max(top_left_x, 0)
            top_left_y = max(top_left_y, 0)
            bottom_right_x = min(bottom_right_x, output_image.shape[1])
            bottom_right_y = min(bottom_right_y, output_image.shape[0])

            # Overlay the shirt on the person's body
            try:
                alpha_s = imgShirt[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    output_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x, c] = (
                        alpha_s * imgShirt[:, :, c] + alpha_l * output_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x, c])
            except Exception as e:
                print(f"Overlay error: {e}")

            # Overlay interactive buttons for shirt selection
            overlay_right_x = output_image.shape[1] - imgButtonRight.shape[1] - 10
            overlay_left_x = 10

            output_image[293:293 + imgButtonRight.shape[0], overlay_right_x:overlay_right_x + imgButtonRight.shape[1]] = imgButtonRight
            output_image[293:293 + imgButtonLeft.shape[0], overlay_left_x:overlay_left_x + imgButtonLeft.shape[1]] = imgButtonLeft

            # Implement button interaction for shirt selection
            if lmList[16][1] < 300:
                counterRight += 1
                cv2.ellipse(output_image, (overlay_right_x + overlay_width // 2, 360), (66, 66), 0, 0,
                            counterRight * selectionSpeed, (0, 255, 0), 20)
                if counterRight * selectionSpeed > 360:
                    counterRight = 0
                    if imageNumber < len(listShirts) - 1:
                        imageNumber += 1
            elif lmList[15][1] > 900:
                counterLeft += 1
                cv2.ellipse(output_image, (overlay_left_x + overlay_width // 2, 360), (66, 66), 0, 0,
                            counterLeft * selectionSpeed, (0, 255, 0), 20)
                if counterLeft * selectionSpeed > 360:
                    counterLeft = 0
                    if imageNumber > 0:
                        imageNumber -= 1
            else:
                counterRight = 0
                counterLeft = 0

        # Overlay cart icon and count
        output_image[10:10 + cart_icon.shape[0], screenWidth - 10 - cart_icon.shape[1]:screenWidth - 10] = cart_icon
        cv2.putText(output_image, str(len(cart)), (screenWidth - 35, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show notification
        if notification_text and (time.time() - notification_timer < 2):  # Show for 2 seconds
            cv2.putText(output_image, notification_text, (screenWidth // 2 - 100, screenHeight // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            notification_text = ""

    except Exception as e:
        print(f"Error: {e}")

    # Display the image with overlays
    cv2.imshow("Virtual Try-On", output_image)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
