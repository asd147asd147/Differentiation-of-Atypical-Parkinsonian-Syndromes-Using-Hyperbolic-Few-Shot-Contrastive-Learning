import glob
import os
import math
import numpy as np
import pickle
import cv2
import nibabel as nib
import torch
import random
SEED = 40

def load_volume(patient_type, file_path):
    lowerpercentile, upperpercentile = 0.2, 99.8

    nii = nib.load(os.path.join('Dataset', patient_type, file_path))
    raw_volume = nii.get_fdata().astype('float32')

    if lowerpercentile is not None:
        qlow = np.percentile(raw_volume, lowerpercentile)
    if upperpercentile is not None:
        qup = np.percentile(raw_volume, upperpercentile)

    if lowerpercentile is not None:
        raw_volume[raw_volume < qlow] = qlow
    if upperpercentile is not None:
        raw_volume[raw_volume > qup] = qup
    
    raw_volume = (raw_volume / raw_volume.max())
    raw_volume = raw_volume[40:140,80:160,40:100]
    return raw_volume

class BrainUMAP(object):
    def __init__(self):
        self.train_Subtype = [
            'PD',
            'NC',
            'MSAC',
            'MSAP',
            'PSP'
        ]

        self.train_patient_path = []
        self.train_patient_type = []
        self.train_data_num = 0

        self.query_path = []

        with open('./query_dataset.txt', "r") as f:
            lines = f.readlines()
            for line in lines:
                self.query_path.append(line.strip())


        for i in self.train_Subtype:
            file_path = os.listdir(os.path.join('Dataset', i))
            self.train_data_num += len(file_path)
            self.train_patient_path += file_path
            self.train_patient_type += [i]*len(file_path)

    def __len__(self):
        return self.train_data_num

    def __getitem__(self, index):
        patinet_type = self.train_patient_type[index]
        file_path = self.train_patient_path[index]
        isquery = False
        if os.path.basename(file_path) in self.query_path:
            isquery = True
        volume1 = load_volume(patinet_type, file_path)
        volume1 = np.expand_dims(volume1, axis=0)
        return volume1, patinet_type, isquery, os.path.basename(file_path)

class BrainSupportDataset(object):
    def __init__(self, support_dataset, base_idx):
        self.support_dataset = support_dataset
        self.patient_type = list(support_dataset.keys())
        self.data_paths = []
        self.data_labels = []
        self.base_idx = base_idx

        for key, value in support_dataset.items():
            self.data_paths += value
            self.data_labels += [key] * len(value)

        with open('./query_dataset.txt', "w") as f:
            for data_path in self.data_paths:
                f.write(data_path+"\n")

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        source_patinet_type = self.data_labels[index]
        volume1 = load_volume(source_patinet_type, self.data_paths[index])
        volume1 = np.expand_dims(volume1, axis=0)
        return volume1, self.base_idx+self.patient_type.index(source_patinet_type)


class BrainQueryDataset(object):
    def __init__(self, query_dataset):
        self.query_dataset = query_dataset
        self.patient_type = list(query_dataset.keys())
        self.data_paths = []
        self.data_labels = []

        for key, value in query_dataset.items():
            self.data_paths += value
            self.data_labels += key * len(value)

        with open('./query_dataset.txt', "w") as f:
            for data_path in self.data_paths:
                f.write(data_path+"\n")

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
 
        source_patinet_type = random.choice(self.patient_type)
        target_patient_type = None

        should_get_same_class = random.randint(0,1)

        volume1 = load_volume(source_patinet_type, random.choice(self.query_dataset[source_patinet_type]))
        volume2 = None

        if should_get_same_class:
            volume2 = load_volume(source_patinet_type, random.choice(self.query_dataset[source_patinet_type]))
            target_patient_type = source_patinet_type
        else:
            while True:
                target_patient_type = random.choice(self.patient_type)
                if source_patinet_type != target_patient_type: break
            volume2 = load_volume(target_patient_type, random.choice(self.query_dataset[target_patient_type]))
        volume1 = np.expand_dims(volume1, axis=0)
        volume2 = np.expand_dims(volume2, axis=0)
        return volume1, volume2, torch.from_numpy(np.array([int(not should_get_same_class)], dtype=np.float32))

class BrainDataset(object):
    def __init__(self, k_shot=3, training=True):
        random.seed(SEED)
        self.train_Subtype = [
            'PD',
            'NC',
            'MSAC',
        ]
        self.test_Subtype = [
            'PD',
            'NC',
            'MSAC',
            'MSAP',
            'PSP',
        ]

        self.training = training

        self.train_patient_path = {}
        self.test_query_patient_path = []
        self.test_query_patient_label = []
        self.test_support_patient_path = {}

        self.train_data_num = 0

        self.k_shot = k_shot

        for i in self.train_Subtype:
            file_path = os.listdir(os.path.join('Dataset', i))
            self.train_data_num += len(file_path)
            self.train_patient_path[i] = file_path
        for i in self.test_Subtype:
            file_path = os.listdir(os.path.join('Dataset', i))
            support_set = random.choices(file_path,k=self.k_shot)
            self.test_support_patient_path[i] = support_set
            query_set = set(file_path) - set(support_set)
            self.test_query_patient_path += list(query_set)
            self.test_query_patient_label += [i] * len(query_set)

    def __len__(self):
        if self.training:
            return self.train_data_num
        return len(self.test_query_patient_path)

    def __getitem__(self, index):
        if self.training:
            source_patinet_type = random.choice(self.train_Subtype)
            target_patient_type = None

            should_get_same_class = random.randint(0,1)

            volume1 = load_volume(source_patinet_type, random.choice(self.train_patient_path[source_patinet_type]))
            volume2 = None

            if should_get_same_class:
                volume2 = load_volume(source_patinet_type, random.choice(self.train_patient_path[source_patinet_type]))
                target_patient_type = source_patinet_type
            else:
                while True:
                    target_patient_type = random.choice(self.train_Subtype)
                    if source_patinet_type != target_patient_type: break
                volume2 = load_volume(target_patient_type, random.choice(self.train_patient_path[target_patient_type]))
            volume1 = np.expand_dims(volume1, axis=0)
            volume2 = np.expand_dims(volume2, axis=0)
            return volume1, volume2, torch.from_numpy(np.array([int(not should_get_same_class)], dtype=np.float32))
        else:
            return load_volume(self.test_query_patient_label[index], self.test_query_patient_path[index]), self.test_Subtype.index(self.test_query_patient_label[index])
        

class BrainDatasetHyper(object):
    def __init__(self, k_shot=3, training=True):
        random.seed(SEED)
        self.train_Subtype = [
            'PD',
            'NC',
            'MSAC',
            # 'MSAP',
            # 'PSP',
        ]
        self.test_Subtype = [
            'PD',
            'NC',
            'MSAC',
            'MSAP',
            'PSP',
        ]

        self.training = training

        self.train_patient_path = []    
        self.train_patient_label = []
        self.test_query_patient_path = []
        self.test_query_patient_label = []
        self.test_support_patient_path = {}

        self.train_data_num = 0

        self.k_shot = k_shot

        for i in self.train_Subtype:
            file_path = os.listdir(os.path.join('Dataset', i))
            self.train_data_num += len(file_path)
            self.train_patient_path += file_path
            self.train_patient_label += [i]*len(file_path)
        for i in self.test_Subtype:
            file_path = os.listdir(os.path.join('Dataset', i))
            support_set = random.choices(file_path,k=self.k_shot)
            self.test_support_patient_path[i] = support_set
            query_set = set(file_path) - set(support_set)
            self.test_query_patient_path += list(query_set)
            self.test_query_patient_label += [i] * len(query_set)

    def __len__(self):
        if self.training:
            return self.train_data_num
        return len(self.test_query_patient_path)
    
    def __getitem__(self, index):
        if self.training:
            source_patinet_type = self.train_patient_label[index]
            volume1 = load_volume(source_patinet_type, self.train_patient_path[index])
            volume1 = np.expand_dims(volume1, axis=0)
            return volume1, self.train_Subtype.index(source_patinet_type)
        else:
            return load_volume(self.test_query_patient_label[index], self.test_query_patient_path[index]), self.test_Subtype.index(self.test_query_patient_label[index])