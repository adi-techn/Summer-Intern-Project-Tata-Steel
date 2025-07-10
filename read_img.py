import cv2 as cv
import sys

img=cv.imread(r"C:\Users\Aditya Kumar Singh\OneDrive\Pictures\Doc\Profile.jpg")

if img is None:
     sys.exit("Image not found")
cv.imshow("Profile",img)

cv.waitKey(0)
cv.destroyAllWindows()