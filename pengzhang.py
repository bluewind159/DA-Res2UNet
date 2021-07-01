import cv2
import numpy as np
import matplotlib.pyplot as plt
# Morphology Dilate
def Morphology_Dilate(img, Dil_time=1):
    H, W = img.shape

    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)

    # each dilate time
    out = img.copy()
    for i in range(Dil_time):
        tmp = np.pad(out, (1, 1), 'edge')
        for y in range(1, H):
            for x in range(1, W):
                if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                    out[y, x] = 255

    return out


# Morphology Erode
def Morphology_Erode(img, Erode_time=1):
    H, W = img.shape
    out = img.copy()

    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)

    # each erode
    for i in range(Erode_time):
        tmp = np.pad(out, (1, 1), 'edge')
        # erode
        for y in range(1, H):
            for x in range(1, W):
                if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                    out[y, x] = 0

    return out
    
# Read image
#img = cv2.imread("target.png",cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Otsu's binarization
#otsu = otsu_binarization(img)

# Morphology - dilate
#erode_result = Morphology_Erode(otsu, Erode_time=2)
#dilate_result = Morphology_Dilate(otsu,Dil_time=2)
 
# Save result
#cv2.imwrite("Black_and_white.jpg",otsu)
#cv2.imshow("Black_and_white",otsu)
#cv2.imwrite("erode_result.jpg", erode_result)
#cv2.imshow("erode_result", erode_result)
#cv2.imwrite("dilate_result.jpg", dilate_result)
#cv2.imshow("dilate_result",dilate_result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()