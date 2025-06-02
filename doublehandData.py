import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import os

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Allow detection of up to two hands

offset = 20
imgSize = 300  # Size of the white canvas

# Folder to save images
folder = r"C:\Users\HP\Desktop\DATA2\W"  # Use raw string to handle Windows path
if not os.path.exists(folder):
    os.makedirs(folder)  # Ensure the folder exists

print(f"Saving images to: {folder}")  # Debugging: Check folder path
counter = 0

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    imageWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255

    if hands:
        print(f"Hands detected: {len(hands)}")
        x_min = min(hand['bbox'][0] for hand in hands)
        y_min = min(hand['bbox'][1] for hand in hands)
        x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
        y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)

        # Ensure crop boundaries are valid
        y1, y2 = max(0, y_min - offset), min(img.shape[0], y_max + offset)
        x1, x2 = max(0, x_min - offset), min(img.shape[1], x_max + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("Warning: Empty cropped image, skipping.")
        else:
            # Resize maintaining aspect ratio
            aspectRatio = imgCrop.shape[1] / imgCrop.shape[0]
            if aspectRatio > 1:  # Wider than tall
                newWidth = imgSize
                newHeight = int(imgSize / aspectRatio)
                imgResize = cv2.resize(imgCrop, (newWidth, newHeight))
                hGap = (imgSize - newHeight) // 2
                imageWhite[hGap:hGap + newHeight, :] = imgResize
            else:  # Taller than wide
                newHeight = imgSize
                newWidth = int(imgSize * aspectRatio)
                imgResize = cv2.resize(imgCrop, (newWidth, newHeight))
                wGap = (imgSize - newWidth) // 2
                imageWhite[:, wGap:wGap + newWidth] = imgResize

            # Save image immediately after detection
            counter += 1
            timestamp = time.time()
            file_path = os.path.join(folder, f"ISL_A_{timestamp:.0f}_{counter}.jpg")
            cv2.imwrite(file_path, imageWhite)
            print(f"Saved: {file_path}")
            print(f"Total Images: {counter}")

            cv2.imshow("Detected Hands", imageWhite)

    cv2.imshow("Image", img)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
