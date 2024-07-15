import os
import cv2
from cvzone.PoseModule import PoseDetector
import pyautogui

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

# Manually set shirt size parameters
widthOfShirt = 400  # Set a fixed width for the shirt
shirtRatioHeightWidth = 1.2  # Adjust this ratio to modify the height of the shirt relative to its width

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

# Fixed position for the shirt
fixed_x = 300  # X coordinate
fixed_y = 200  # Y coordinate

while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image from webcam.")
        continue

    try:
        # Detect pose and landmarks
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

        if lmList:
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

            # Manually set shirt size
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))

            # Fixed coordinates for the shirt
            shirt_height, shirt_width, _ = imgShirt.shape
            top_left_x = fixed_x
            top_left_y = fixed_y
            bottom_right_x = top_left_x + shirt_width
            bottom_right_y = top_left_y + shirt_height

            # Ensure coordinates are within the image boundaries
            top_left_x = max(top_left_x, 0)
            top_left_y = max(top_left_y, 0)
            bottom_right_x = min(bottom_right_x, img.shape[1])
            bottom_right_y = min(bottom_right_y, img.shape[0])

            # Overlay the shirt on the person's body
            try:
                overlay_shirt = imgShirt[:, :, :3] * (imgShirt[:, :, 3:] / 255.0) + img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] * (1 - imgShirt[:, :, 3:] / 255.0)
                img[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = overlay_shirt.astype('uint8')
            except Exception as e:
                print(f"Overlay error: {e}")

            # Overlay interactive buttons for shirt selection
            overlay_right_x = img.shape[1] - imgButtonRight.shape[1] - 10
            overlay_left_x = 10

            img[293:293 + imgButtonRight.shape[0], overlay_right_x:overlay_right_x + imgButtonRight.shape[1]] = imgButtonRight
            img[293:293 + imgButtonLeft.shape[0], overlay_left_x:overlay_left_x + imgButtonLeft.shape[1]] = imgButtonLeft

            # Implement button interaction for shirt selection
            if lmList[16][1] < 300:
                counterRight += 1
                cv2.ellipse(img, (139, 360), (66, 66), 0, 0,
                            counterRight * selectionSpeed, (0, 255, 0), 20)
                if counterRight * selectionSpeed > 360:
                    counterRight = 0
                    if imageNumber < len(listShirts) - 1:
                        imageNumber += 1
            elif lmList[15][1] > 900:
                counterLeft += 1
                cv2.ellipse(img, (1138, 360), (66, 66), 0, 0,
                            counterLeft * selectionSpeed, (0, 255, 0), 20)
                if counterLeft * selectionSpeed > 360:
                    counterLeft = 0
                    if imageNumber > 0:
                        imageNumber -= 1
            else:
                counterRight = 0
                counterLeft = 0

    except Exception as e:
        print(f"Error: {e}")

    # Display the image with overlays
    cv2.imshow("Virtual Try-On", img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
