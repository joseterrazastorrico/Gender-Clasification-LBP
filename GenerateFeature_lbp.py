
from pyClass.FeatureExtraction import LocalBinaryPattern
from skimage.color import rgb2gray
from skimage import io
from tqdm import tqdm
import pandas as pd
import os

def main():

	dir_input = input("Ingrese el directorio donde se encuentra la data: \n")
	if dir_input=="": dir_input = "./GenderDataProcessed"
	dir_output = input("Ingrese el directorio donde se guarda el output: \n")
	if dir_output=="": dir_output = "./DataFeatures"
	method = input("Ingrese metodo de LBP ('uniform' o 'nri_uniform'): \n")
	if method=="": method = "nri_uniform"

	lbp = LocalBinaryPattern(8, 1, method=method)
	for carp in os.listdir(dir_input):
		datas = []
		for cat in os.listdir(os.path.join(dir_input, carp)):
			data = []
			lista_img_cat = os.listdir(os.path.join(dir_input, carp, cat))
			for i in tqdm(range(len(lista_img_cat)), "Progreso {}-{}".format(carp, cat)):
				img = rgb2gray(io.imread(os.path.join(dir_input, carp, cat,lista_img_cat[i])))
				data.append(lbp.describe(img))
			data = pd.DataFrame(data)
			data.columns = ['V{}'.format(i) for i in range(data.columns.shape[0])]
			data['y'] = cat
			datas.append(data)
		datas = pd.concat(datas, axis=0).reset_index(drop=True)
		datas.to_csv(dir_output+'/FeaturesData_{}_{}.csv'.format(method, carp), index=False)

	return 0

if __name__ == "__main__":
    main()