import pickle

def convert(filename):
    data=[]
    with open(filename,"r") as F:
        X,y=[],[]
        for line in F:
            if len(line)<=1:
                data.append([X,y])
                X,y=[],[]
                continue
            #print(len(line))
            X.append(line.split("\t")[0])
            y.append(line.split("\t")[3][:-1])
    with open("NCBI_data_"+filename+".pkl","wb")as F1:
        pickle.dump(data,F1,protocol=3)

convert("train.txt")
convert("test.txt")
convert("dev.txt")