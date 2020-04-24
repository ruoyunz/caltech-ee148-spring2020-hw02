import os
import numpy as np
import json
from PIL import Image, ImageDraw

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../data/hw02_preds' 
os.makedirs(preds_path, exist_ok=True) # create directory if needed 

def main():
	# get sorted list of files: 
	file_names = sorted(os.listdir(data_path)) 
	# remove any non-JPEG files: 
	file_names = [f for f in file_names if '.jpg' in f] 

	with open(os.path.join(preds_path,'preds_train.json')) as f:
	    boxes = json.load(f)

	for i in range(len(file_names)):
		if file_names[i] in boxes:
		    I = Image.open(os.path.join(data_path,file_names[i]))
		    I = np.asarray(I)
		    img = Image.fromarray(I)
		    draw = ImageDraw.Draw(img)

		    preds = boxes[file_names[i]]
		    print(preds)
		    for p in preds:
		    	draw.rectangle((p[1], p[0], p[3], p[2]), outline=(0, 255, 0))

		    img.save(os.path.join(data_path,"../hw02_preds/pred_" + file_names[i] + ".jpg"))

if __name__ == "__main__":
    main()