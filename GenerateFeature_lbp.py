
from pyClass.FeatureExtraction import LocalBinaryPattern
from skimage.color import rgb2gray
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os


def dividir_imagen(img):
	x,y = img.shape
	img1 = img[0:int(x/2), 0:int(y/2)]
	img2 = img[0:int(x/2), int(y/2):y]
	img3 = img[int(x/2):x, 0:int(y/2)]
	img4 = img[int(x/2):x, int(y/2):y]
	return img1, img2, img3, img3

def main():

	dir_input = input("Ingrese el directorio donde se encuentra la data: \n")
	if dir_input=="": dir_input = "./GenderDataProcessed"
	dir_output = input("Ingrese el directorio donde se guarda el output: \n")
	if dir_output=="": dir_output = "./DataFeatures"
	method = input("Ingrese metodo de LBP ('uniform' o 'nri_uniform'): \n")
	if method=="": method = "nri_uniform"
	improved = input("Desea la version mejorada (si:'y', no:'n'): \n")
	if method=="": method = "y"

	lbp = LocalBinaryPattern(8, 1, method=method)
	for carp in os.listdir(dir_input):
		datas = []
		for cat in os.listdir(os.path.join(dir_input, carp)):
			data = []
			lista_img_cat = os.listdir(os.path.join(dir_input, carp, cat))
			for i in tqdm(range(len(lista_img_cat)), "Progreso {}-{}".format(carp, cat)):
				img = rgb2gray(io.imread(os.path.join(dir_input, carp, cat,lista_img_cat[i])))
				if improved=='y':
					img1, img2, img3, img4 = dividir_imagen(img)
					lbp_imagen = np.ravel([lbp.describe(img1),lbp.describe(img2),lbp.describe(img3),lbp.describe(img4)])
				else:
					lbp_imagen(lbp.describe(img))
				data.append(lbp_imagen)
			data = pd.DataFrame(data)
			data.columns = ['V{}'.format(i) for i in range(data.columns.shape[0])]
			data['y'] = cat
			datas.append(data)
		datas = pd.concat(datas, axis=0).reset_index(drop=True)
		if improved=='y':

			
			datas.to_csv(dir_output+'/FeaturesData_improved_{}_{}.csv'.format(method, carp), index=False)
		else:
			datas.to_csv(dir_output+'/FeaturesData_{}_{}.csv'.format(method, carp), index=False)

	return 0

if __name__ == "__main__":
    main()