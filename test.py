from main import modelMethod
# res = modelMethod['predict']('./data/12549/0.jpg')
import cv2
img = cv2.imread('./data/12549/0.jpg')
res = modelMethod['predict'](img)
print(res)