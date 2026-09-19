import os
import numpy as np
import h5py

import scipy.io as scio

data_path = '../mpi_inf_3dhp/test_data'
cam_set = [0, 1, 2, 4, 5, 6, 7, 8]
# # H36M-compatible joint order: [pelvis, rhip, rknee, rankle, lhip, lknee, lankle, spine, neck, head_top, head, lsho, lelb, lwri, rsho, relb, rwri]
# joint_set = [4, 23, 24, 25, 18, 19, 20, 3, 5, 7, 6, 9, 10, 11, 14, 15, 16]

joint_set = [4, 18, 19, 20, 23, 24, 25, 3, 5, 7, 6, 14, 15, 16, 9, 10, 11]
dic_seq={}

for root, dirs, files in os.walk(data_path):

    for file in files:
        if file.endswith("mat"):

            path = root.split("/")
            subject = path[-1][2]
            print("loading %s..."%path[-1])

            data = h5py.File(os.path.join(root, file))

            valid_frame = np.squeeze(data['valid_frame'][:])

            data_2d = np.squeeze(data['annot2'][:])
            data_3d = np.squeeze(data['univ_annot3'][:])

            # Reorder from MPI CPM order to H36M order
            # MPI CPM: [head_top, neck, rsho, relb, rwri, lsho, lelb, lwri, rhip, rknee, rankle, lhip, lknee, lankle, pelvis, spine, head]
            # H36M:    [pelvis, rhip, rknee, rankle, lhip, lknee, lankle, spine, neck, head_top, head, lsho, lelb, lwri, rsho, relb, rwri]
            reorder = [14, 11, 12, 13, 8, 9, 10, 15, 1, 0, 16, 2, 3, 4, 5, 6, 7]
            data_2d = data_2d[:, reorder, :]
            data_3d = data_3d[:, reorder, :]

            dic_data = {"data_2d":data_2d,"data_3d":data_3d, "valid":valid_frame}

            dic_seq.update({path[-1]:dic_data})

np.savez_compressed('../motion3d/data_test_3dhp', data=dic_seq)
