import os 
import shutil
f = open("/data/changty/DG-segment-code/RobustNet/split_data/gtav_split_train.txt")               
# lines = f.readlines() 

lines = os.listdir('/data/changty/DG-segment-dataset/gtav/images')
a = 0
for line in lines:
    os.mkdir('/data/changty/DG-segment-dataset/gtav/image/train/'+line[0:5])
    os.mkdir('/data/changty/DG-segment-dataset/gtav/label/train/'+line[0:5])
    shutil.copy(("/data/changty/DG-segment-dataset/gtav/images/" + line), '/data/changty/DG-segment-dataset/gtav/image/train/'+line[0:5])
    shutil.copy(("/data/changty/DG-segment-dataset/gtav/labels/" + line), '/data/changty/DG-segment-dataset/gtav/label/train/'+line[0:5])