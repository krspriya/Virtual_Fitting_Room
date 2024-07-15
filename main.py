import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector

# Access the camera
cap = cv2.VideoCapture(0)
detector = PoseDetector()

shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)
fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 581 / 440
selectedShirtIndex = -1  # No shirt selected initially

# Load shirt images with error handling
shirtImages = []
for shirt in listShirts:
    shirtPath = os.path.join(shirtFolderPath, shirt)
    if os.path.isfile(shirtPath):
        img = cv2.imread(shirtPath, cv2.IMREAD_UNCHANGED)
        if img is not None:
            shirtImages.append(img)
        else:
            print(f"Error loading image: {shirtPath}")
    else:
        print(f"File not found: {shirtPath}")

def displayShirtOptions(img):
    y_offset = 10
    for idx, shirt in enumerate(shirtImages):
        thumbnail = cv2.resize(shirt, (100, 100), interpolation=cv2.INTER_AREA)
        x_offset = 10 + (idx * 110)
        overlay_image_with_alpha(img, thumbnail, (x_offset, y_offset))
        cv2.rectangle(img, (x_offset, y_offset), (x_offset + thumbnail.shape[1], y_offset + thumbnail.shape[0]), (0, 255, 0), 2)
    return img

def overlay_image_with_alpha(background, overlay, position):
    x, y = position

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(overlay)
    overlay_color = cv2.merge((b, g, r))

    # Apply the overlay
    mask = cv2.merge((a, a, a))
    inv_mask = cv2.bitwise_not(mask)

    try:
        roi = background[y:y+overlay.shape[0], x:x+overlay.shape[1]]
        roi_bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
        roi_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)
        combined = cv2.add(roi_bg, roi_fg)
        background[y:y+overlay.shape[0], x:x+overlay.shape[1]] = combined
    except:
        pass

def isInsideRectangle(x, y, x1, y1, x2, y2):
    return x1 < x < x2 and y1 < y < y2

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1280, 720))  # Resize image for better visibility
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

    # Display shirt options on screen
    img = displayShirtOptions(img)

    if lmList:
        lm11 = lmList[11][1:3]
        lm12 = lmList[12][1:3]

        if selectedShirtIndex >= 0:
            imgShirt = shirtImages[selectedShirtIndex]

            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)

            if widthOfShirt > 0:  # Check if width is positive
                imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)), interpolation=cv2.INTER_AREA)
                currentScale = (lm11[0] - lm12[0]) / 190
                offset = int(44 * currentScale), int(48 * currentScale)

                try:
                    overlay_image_with_alpha(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
                except Exception as e:
                    print(f"Error overlaying shirt: {e}")

    cv2.imshow("Image", img)
    
    # Check for mouse click to select shirt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:        
        break

    def onMouse(event, x, y, flags, param):
        global selectedShirtIndex
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx in range(len(shirtImages)):
                x1 = 10 + (idx * 110)
                y1 = 10
                x2 = x1 + 100
                y2 = y1 + 100
                if isInsideRectangle(x, y, x1, y1, x2, y2):
                    selectedShirtIndex = idx

    cv2.setMouseCallback("Image", onMouse)

cap.release()
cv2.destroyAllWindows()
