import cv2
import os

path = "./labels/"

for d1 in os.listdir(path):
    r1 = path + d1
    try:
        for d2 in os.listdir(r1):
            r2 = r1 + '/' + d2
            for d3 in os.listdir(r2):
                r3 = r2 + '/' + d3
                image = cv2.imread(r3)
                src = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_AREA)
                cv2.imwrite(r3, src)
                cv2.waitKey(0)
    except NotADirectoryError:
        pass

cv2.destroyAllWindows()