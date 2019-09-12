import os
import numpy as np

RUN = "/home/leon/dev/CSN_showcase/CSN_chexpert/RUNS_ensamble/20190911-14:22/"

#atel_ = list()
#cardiomegaly_ = list()
#consol_ = list()
#edema_ = list()
#pe_ = list()

dct = dict()


for root, dirs, files in os.walk(RUN):
    for dir in dirs:
        if dir.startswith("epoch"):
            with open(RUN + dir +"/performance.txt" ) as f:
                lst = f.readlines()
                lst = lst[:5]
                #print(lst)
                for row in lst:
                    exec("entry = %s" % row)

                    try:
                        dct[entry[0]].append(entry[1])
                    except:
                        dct[entry[0]] = list()
                        dct[entry[0]].append(entry[1])

for key in dct:
    arr = np.array(dct[key])
    indices = np.argsort(dct[key])
    inds = indices[-10:]
    #print(key ,arr[inds])

    print("\n\n%s AVERAGE AUC: %s SD: %s MIN: %s MAX: %s" %(key, np.mean(arr[inds]), np.std(arr[inds]), np.min(arr[inds]), np.max(arr[inds])))

