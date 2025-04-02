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
import random

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
FASTA_FILE = "hg38.fa"

WORK_DIR = "./"


def get_chrom2seq(FASTA_FILE, capitalize=True):

    chrom2seq = {}
    for seq in SeqIO.parse(FASTA_FILE, "fasta"):
        chrom2seq[seq.description.split()[0]] = seq.seq.upper() if capitalize else seq.seq

    return chrom2seq


def seq2one_hot(seq):

    d = np.array(['A', 'C', 'G', 'T'])

    return np.fromstring(str(seq.upper()), dtype='|S1')[:, np.newaxis] == d


def get_models():

    phase_one_model_file = os.path.join(WORK_DIR, "source_files/phase_one_model.hdf5")
    phase_one_weights_file = os.path.join(WORK_DIR, "source_files/phase_one_weights.hdf5")
    phase_two_model_file = os.path.join(WORK_DIR, "source_files/phase_two_model.hdf5")
    phase_two_weights_file = os.path.join(WORK_DIR, "source_files/phase_two_weights.hdf5")

    model_1 = load_model(phase_one_model_file)
    model_1.load_weights(phase_one_weights_file)
    model_2 = load_model(phase_two_model_file)
    model_2.load_weights(phase_two_weights_file)

    return [model_1, model_2]


def get_phase_one_model():

    phase_one_model_file = os.path.join(WORK_DIR, "source_files/phase_one_model.hdf5")
    phase_one_weights_file = os.path.join(WORK_DIR, "source_files/phase_one_weights.hdf5")

    model = load_model(phase_one_model_file)
    model.load_weights(phase_one_weights_file)

    return model

def create_dataset_phase_two_unbinned(positive_bed_file, negative_bed_file, dataset_save_file, FASTA_FILE, chrom2seq=None,
                                      model=None):

    if not chrom2seq:
        chrom2seq = get_chrom2seq(FASTA_FILE)

    if not model:
        model = get_phase_one_model()

    print "Generating the positive dataset"

    pos_beds = list(BedTool(positive_bed_file))
    for r in pos_beds:
        c = int((r.start+r.stop)/2)
        r.start = c-1000
        r.stop = c+1000

    neg_beds = list(BedTool(negative_bed_file))
    for r in neg_beds:
        c = int((r.start+r.stop)/2)
        r.start = c-1000
        r.stop = c+1000
    
    poslab = np.genfromtxt(positive_bed_file+".label",delimiter="\t")

    pos_train_bed = [r for r in pos_beds if r.chrom in train_chromosomes]
    pos_val_bed = [r for r in pos_beds if r.chrom in validation_chromosomes]
    pos_test_bed = [r for r in pos_beds if r.chrom in test_chromosomes]
    pos_train_v = [r2 for r1,r2 in zip(pos_beds,poslab) if r1.chrom in train_chromosomes]
    pos_val_v = [r2 for r1,r2 in zip(pos_beds,poslab) if r1.chrom in validation_chromosomes]
    pos_test_v = [r2 for r1,r2 in zip(pos_beds,poslab) if r1.chrom in test_chromosomes]
    print(len(pos_train_v), len(pos_val_v),len(pos_train_bed))

    pos_train_data = []
    pos_val_data = []
    pos_test_data = []

    pos_train_label = []
    pos_val_label = []
    pos_test_label = []
    for bed_list, data_list, label_list, vv in zip([pos_train_bed, pos_val_bed, pos_test_bed],
                                   [pos_train_data, pos_val_data, pos_test_data],
				    [pos_train_label,pos_val_label,pos_test_label],
                                    [pos_train_v, pos_val_v,pos_test_v]):

        for r,b in zip(bed_list,vv):
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == 2000:
                continue
            _vector = seq2one_hot(_seq)

            b1 = 0
            if b[0]>0:
                b1=1
            if b[1]>0:
                b1=-1
            if b1 == 0:
                continue
            if (b1==1) & (b[2:].sum()==0):
                continue

            sk = np.zeros((22,))
            sk[0] = b1
            sk[1:] = b[2:]

            data_list.append(_vector)
            label_list.append(sk)

    print len(pos_train_data)
    print len(pos_val_data)
    print len(pos_test_data)

    print "Generating the negative dataset"

    neg_train_bed = [r for r in neg_beds if r.chrom in train_chromosomes]
    neg_val_bed = [r for r in neg_beds if r.chrom in validation_chromosomes]
    neg_test_bed = [r for r in neg_beds if r.chrom in test_chromosomes]

    neg_train_data = []
    neg_val_data = []
    neg_test_data = []

    for bed_list, data_list in zip([neg_train_bed, neg_val_bed, neg_test_bed],
                                   [neg_train_data, neg_val_data, neg_test_data]):
        # for bed_list, data_list in zip([neg_test_bed], [neg_test_data]):
        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == 2000:
                continue

            _vector = seq2one_hot(_seq)
            data_list.append(_vector)

    print len(neg_train_data)
    print len(neg_val_data)
    print len(neg_test_data)

    print "Merging positive and negative to single matrices"

    pos_train_data_matrix = np.zeros((len(pos_train_data), INPUT_LENGTH, 4))
    for i in range(len(pos_train_data)):
        pos_train_data_matrix[i, :, :] = pos_train_data[i]
    pos_val_data_matrix = np.zeros((len(pos_val_data), INPUT_LENGTH, 4))
    for i in range(len(pos_val_data)):
        pos_val_data_matrix[i, :, :] = pos_val_data[i]
    pos_test_data_matrix = np.zeros((len(pos_test_data), INPUT_LENGTH, 4))
    for i in range(len(pos_test_data)):
        pos_test_data_matrix[i, :, :] = pos_test_data[i]

    neg_train_data_matrix = np.zeros((len(neg_train_data), INPUT_LENGTH, 4))
    for i in range(len(neg_train_data)):
        neg_train_data_matrix[i, :, :] = neg_train_data[i]
    neg_val_data_matrix = np.zeros((len(neg_val_data), INPUT_LENGTH, 4))
    for i in range(len(neg_val_data)):
        neg_val_data_matrix[i, :, :] = neg_val_data[i]
    neg_test_data_matrix = np.zeros((len(neg_test_data), INPUT_LENGTH, 4))
    for i in range(len(neg_test_data)):
        neg_test_data_matrix[i, :, :] = neg_test_data[i]

    test_data = np.vstack((pos_test_data_matrix, neg_test_data_matrix))
    train_data = np.vstack((pos_train_data_matrix, neg_train_data_matrix))
    val_data = np.vstack((pos_val_data_matrix, neg_val_data_matrix))
    
    test_label = np.concatenate((np.array(pos_test_label), np.zeros((neg_test_data_matrix.shape[0],22))))
    train_label = np.concatenate((np.array(pos_train_label), np.zeros((neg_train_data_matrix.shape[0],22))))
    val_label = np.concatenate((np.array(pos_val_label), np.zeros((neg_val_data_matrix.shape[0],22))))
    print((train_label==-1).sum(), (train_label==0).sum(), (train_label==1).sum())
    print((val_label==-1).sum(), (val_label==0).sum(), (val_label==1).sum())
    print((test_label==-1).sum(), (test_label==0).sum(), (test_label==1).sum())
    
    test_data = model.predict(test_data)
    train_data = model.predict(train_data)
    val_data = model.predict(val_data)

    print "Saving to file:", dataset_save_file

    with h5py.File(dataset_save_file, "w") as of:
        of.create_dataset(name="test_data", data=test_data, compression="gzip")
        of.create_dataset(name="train_data", data=train_data, compression="gzip")
        of.create_dataset(name="val_data", data=val_data, compression="gzip")
        of.create_dataset(name="test_label", data=test_label, compression="gzip")
        of.create_dataset(name="train_label", data=train_label, compression="gzip")
        of.create_dataset(name="val_label", data=val_label, compression="gzip")

def create_dataset(posf, negf, fastaf):

    chrom2seq = get_chrom2seq(fastaf)

    model = get_phase_one_model()
    dataset_save_file = "example/"+posf+".phase_one.hdf5"
    create_dataset_phase_two_unbinned(posf, negf, dataset_save_file, fastaf, chrom2seq=chrom2seq, model=model)


def load_dataset(data_file):

    data = {}

    with h5py.File(data_file, "r") as inf:
        for _key in inf:
            data[_key] = inf[_key][()]

    data["train_data"] = data["train_data"][..., np.newaxis]
    data["test_data"] = data["test_data"][..., np.newaxis]
    data["val_data"] = data["val_data"][..., np.newaxis]

    return data


if __name__ == "__main__":

    import sys
    pos_file = sys.argv[1]
    neg_file = sys.argv[2]
    FASTA_file = sys.argv[3]
    create_dataset(pos_file, neg_file, FASTA_file)
