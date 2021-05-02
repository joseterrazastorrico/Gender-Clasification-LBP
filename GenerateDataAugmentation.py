from pyClass.PrepareData import PrepareData

def main():

	dir_input = input("Ingrese el directorio donde se encuentra la data: \n")
	if dir_input=="": dir_input = "./DataInput"
	dir_output = input("Ingrese el nombre del directorio donde estará el output: \n")
	if dir_output=="": 	dir_output = "./GenderDataProcessed"
	data_processing = PrepareData(dir_input, dir_output, cats=[], Train={}, Test={})
	data_processing.TrainTestSplit(p=0.7)
	data_processing.DataAugmentation_all()
	return 0

if __name__ == "__main__":
    main()