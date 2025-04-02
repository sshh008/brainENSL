import os
import numpy as np
import random
import time
import glob
from Bio import SeqIO
from pybedtools import BedTool
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from sklearn import metrics
import h5py
train_chromosomes = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr10", "chr11", "chr12", "chr13",
                     "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22","chrX"]
validation_chromosomes = ["chr7"]
test_chromosomes = ["chr8", "chr9"]

BIN_LENGTH = 200
INPUT_LENGTH = 2000
EPOCH = 200
BATCH_SIZE = 64
GPUS = 4

nucleotides = ['A', 'C', 'G', 'T']

WORK_DIR = "./"


def run_model(data, model, save_dir):
    weights_file = os.path.join(save_dir, "model_weights.hdf5")
    model_file = os.path.join(save_dir, "single_model.hdf5")

    model.save(model_file)

    opt = Adadelta()

    X_train = data["train_data"]
    Y_train = data["train_label"]
    X_validation = data["val_data"]
    Y_validation = data["val_label"]
    X_test = data["test_data"]
    Y_test = data["test_label"]

    i1 = (Y_train[:,0]==-1).sum()
    i2 = (Y_train[:,0]==1).sum()
    if i1 < i2:
	i1 = i2
    cfpos = i1

    i1 = (Y_validation[:,0]==-1).sum()
    i2 = (Y_validation[:,0]==1).sum()
    if i1 < i2:
        i1 = i2
    cfpos1 = i1

    f1 = np.where(np.abs(Y_train[:,0])==1)[0]
    f2 = np.where(Y_train[:,0]==0)[0]
    import random
    random.shuffle(f2)
    f2 = f2[:cfpos*15]
    tid = np.concatenate((f1,f2))

    f1 = np.where(np.abs(Y_validation[:,0])==1)[0]
    f2 = np.where(Y_validation[:,0]==0)[0]
    random.shuffle(f2)
    f2 = f2[:cfpos1*15]
    vid = np.concatenate((f1,f2))

    from keras.utils.np_utils import to_categorical
    from sklearn.utils import class_weight
    
    print(Y_train.mean(axis=0), Y_train.shape)
    print(Y_test.mean(axis=0), Y_test.shape)   
    print(Y_validation.mean(axis=0), Y_validation.shape)
   

    Y_train1 = Y_train[:,0]+1
    Y_test1 = Y_test[:,0]+1
    Y_validation1 = Y_validation[:,0]+1

    i1 = Y_train[:,1:].copy()
    i2 = i1[:,:8].copy()
    i2[:,[2,4,5,6,7,1,0]] = i2[:,[2,4,5,6,7,1,0]] + i1[:,8:15]
    i2[:,[2,4,5,7,0,6]] = i2[:,[2,4,5,7,0,6]] + i1[:,15:]
    Y_traini = (i2>0)+0
    
    i1 = Y_test[:,1:].copy()
    i2 = i1[:,:8].copy()
    i2[:,[2,4,5,6,7,1,0]] = i2[:,[2,4,5,6,7,1,0]] + i1[:,8:15]
    i2[:,[2,4,5,7,0,6]] = i2[:,[2,4,5,7,0,6]] + i1[:,15:]
    Y_testi = (i2>0)+0

    i1 = Y_validation[:,1:].copy()
    i2 = i1[:,:8].copy()
    i2[:,[2,4,5,6,7,1,0]] = i2[:,[2,4,5,6,7,1,0]] + i1[:,8:15]
    i2[:,[2,4,5,7,0,6]] = i2[:,[2,4,5,7,0,6]] + i1[:,15:]
    Y_validationi = (i2>0)+0

    Y_train2 = Y_train[:,0][:,None] * Y_traini+1
    Y_test2 = Y_test[:,0][:,None] * Y_testi+1
    Y_validation2 = Y_validation[:,0][:,None] * Y_validationi+1
   
    model.compile(loss={'class':'sparse_categorical_crossentropy','subclass1':'sparse_categorical_crossentropy',
        'subclass2':'sparse_categorical_crossentropy','subclass3':'sparse_categorical_crossentropy',
        'subclass4':'sparse_categorical_crossentropy', 'subclass5':'sparse_categorical_crossentropy',
        'subclass6':'sparse_categorical_crossentropy', 'subclass7':'sparse_categorical_crossentropy',
        'subclass8':'sparse_categorical_crossentropy'},
        loss_weights={'class':1, 'subclass1':0.5,'subclass2':0.5,'subclass3':0.5,'subclass4':0.5,
            'subclass5':0.5,'subclass6':0.5,'subclass7':0.5,'subclass8':0.5},
        optimizer=opt, 
        metrics=["accuracy"])
    
    _callbacks = []
    checkpoint = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
    _callbacks.append(checkpoint)
    earlystopping = EarlyStopping(monitor="val_loss", patience=15)
    _callbacks.append(earlystopping)

    model.fit(X_train[tid],
            [Y_train1[tid],Y_train2[tid,0],Y_train2[tid,1],Y_train2[tid,2],Y_train2[tid,3],Y_train2[tid,4],Y_train2[tid,5],Y_train2[tid,6],
                Y_train2[tid,7]],
            batch_size=BATCH_SIZE * GPUS,
            epochs=EPOCH,
            validation_data=(X_validation[vid], [Y_validation1[vid],Y_validation2[vid,0],
                Y_validation2[vid,1],Y_validation2[vid,2],Y_validation2[vid,3],Y_validation2[vid,4],Y_validation2[vid,5],Y_validation2[vid,6],
                Y_validation2[vid,7]]),
            #class_weight = labels_dict,
            shuffle=True,
            callbacks=_callbacks, verbose=1)

    model.load_weights(weights_file) 
    Y_pred = model.predict(X_test)
    
    test_result_file = os.path.join(save_dir, "testresult.hdf5")
    with h5py.File(test_result_file, "w") as of:
        of.create_dataset(name="pred", data=Y_pred, compression="gzip")
        of.create_dataset(name="label", data=Y_test, compression="gzip")

    auc1 = metrics.roc_auc_score((Y_test1==0)+0, Y_pred[0][:,0])
    auc2 = metrics.roc_auc_score((Y_test1==2)+0, Y_pred[0][:,2])

    with open(os.path.join(save_dir, "auc.txt"), "w") as of:
        of.write("enhancer AUC: %f\n" % auc2)
        of.write("silencer AUC: %f\n" % auc1) 


def load_dataset2L(data_file):

    print("loading data")
    data = {}
    with h5py.File(data_file, "r") as inf:
        for _key in inf:
            data[_key] = inf[_key][()]

    data["train_data"] = data["train_data"][..., np.newaxis]
    data["test_data"] = data["test_data"][..., np.newaxis]
    data["val_data"] = data["val_data"][..., np.newaxis]

    return data


def train_model(posf, results_dir):

    phase_two_model_file = "./source_files/model_8celltype.hdf5"
    model = load_model(phase_two_model_file)
    if not os.path.exists(posf):
	print("no data file "+posf)
	exit()

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    data = load_dataset2L(posf)
    run_model(data, model, results_dir)


if __name__ == "__main__":

    import sys
    result_dir = "./results.tmp/"

    data_file = "./example/normal.temp.bed.phase_one.hdf5"
    train_model(data_file, result_dir)
