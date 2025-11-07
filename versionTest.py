import cv2

print(cv2.__version__)

color = 1
img = cv2.imread('Cathy Portillo Profile.png', color)
cv2.imshow('Cathy', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('Cathy Portillo Output.png', img)