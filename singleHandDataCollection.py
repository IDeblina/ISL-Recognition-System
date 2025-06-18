import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import os

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300  # Size of the white canvas

# Folder to save images
folder = r"DATA\9"  # Use raw string to handle Windows path
if not os.path.exists(folder):
    os.makedirs(folder)  # Ensure the folder exists

print(f"Saving images to: {folder}")  # Debugging: Check folder path
counter = 0


while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white canvas
        imageWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255

        # Crop the hand region, ensuring boundaries don't exceed the image dimensions
        imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset), 
                      max(0, x - offset):min(img.shape[1], x + w + offset)]

        # Get dimensions of the cropped image
        imgCropShape = imgCrop.shape

        # Resize the cropped image to fit in the white canvas while maintaining aspect ratio
        aspectRatio = imgCropShape[1] / imgCropShape[0]  # width / height

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

        # Display the results
        cv2.imshow("ImageCropped", imgCrop)
        cv2.imshow("ImageWhite", imageWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) 

    if key == ord("s"):
        counter += 1
        # Construct the file path with time-based naming
        file_path = os.path.join(folder, f"Image_{time.time():.0f}_{counter}.jpg")
        print(f"Saving image at: {file_path}")  # Debugging: Check the save path
        cv2.imwrite(file_path, imageWhite)
        print(f"Saved: {file_path}, Total Images: {counter}")

    # Press 'q' to quit the program
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
