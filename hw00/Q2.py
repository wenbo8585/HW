# Q2. 輸入一張圖，將圖上下顛倒，左右相反（旋轉180度），並輸出到ans2.png
import sys
from PIL import Image
import numpy as np

input_file = str(sys.argv[1])

img = Image.open(input_file)
data = np.array(img)
#rotated_data = np.flip(np.flip(data,axis=0),axis=1)
rotated_data = np.flip(data,axis=0)
new_im = Image.fromarray(rotated_data)
new_im.save("ans2.png")

