################################################################################
# Copyright 2019-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# File utility functions
################################################################################

import os
import sys
import csv
import h5py as h5
import torch
import numpy as np

from torch.utils.data import Dataset


# Note: if csv_path exists, the following columns should be included:
#pdb_id/compound_id, pose_id, rmsd, affinity
# if csv doesn't exist, rmsd and affinity become unknown -> only testing is available (no evaluation)

class Dataset_MLHDF(Dataset):
	def __init__(self, mlhdf_path, mlhdf_ver, csv_path="", is_crystal=False, rmsd_weight=False, rmsd_thres=2, max_atoms=2000, feat_dim=22):
		super(Dataset_MLHDF, self).__init__()
		self.mlhdf_ver = mlhdf_ver
		self.mlhdf_path = mlhdf_path
		self.csv_path = csv_path
		self.is_crystal = is_crystal
		self.rmsd_weight = rmsd_weight
		self.rmsd_thres = rmsd_thres
		self.max_atoms = max_atoms
		self.feat_dim = feat_dim

		self.mlhdf = h5.File(self.mlhdf_path, 'r')
		self.data_info_list = []
		if self.mlhdf_ver == 1: # for fusion model
			try:
				with open(self.csv_path, 'r') as fp:
					csv_reader = csv.reader(fp, delimiter=',')
					next(csv_reader)
					for row in csv_reader:
						if float(row[2]) <= rmsd_thres:
							self.data_info_list.append([row[0], row[1], float(row[2])])
			except IOError:
				for comp_id in self.mlhdf.keys():
					if self.is_crystal:
						self.data_info_list.append([comp_id, 0, 0])
					else:
						pose_ids = self.mlhdf[comp_id]["pybel"]["processed"]["docking"].keys()
						for pose_id in pose_ids:
							self.data_info_list.append([comp_id, pose_id, 0, 0])
		elif self.mlhdf_ver == 1.5: # for cfusion model
			if is_crystal:
				for pdbid in self.mlhdf["regression"].keys():
					affinity = float(self.mlhdf["regression"][pdbid].attrs["affinity"])
					self.data_info_list.append([pdbid, 0, 0, affinity])
			else:
				print("not supported!")

	def close(self):
		self.mlhdf.close()

	def __len__(self):
		count = len(self.data_info_list)
		return count

	def __getitem__(self, idx):
		pdbid, poseid, affinity = self.data_info_list[idx]
		data = np.zeros((self.max_atoms, self.feat_dim), dtype=np.float32)
		if self.mlhdf_ver == 1:
			if self.is_crystal:
				mlhdf_ds = self.mlhdf[str(pdbid)]["pybel"]["processed"]['pdbbind']
			else:
				mlhdf_ds = self.mlhdf[pdbid]["pybel"]["processed"]["docking"][poseid]
			actual_data = mlhdf_ds["data"][:]
			data[:actual_data.shape[0],:] = actual_data
		elif self.mlhdf_ver == 1.5:
			if self.is_crystal:
				mlhdf_ds = self.mlhdf["regression"][pdbid]["pybel"]["processed"]
				# the one in ["pdbbind_3dcnn"] is the actual 19x48x48x48
				actual_data = mlhdf_ds["pdbbind_sgcnn"]["data0"][:]
			data[:actual_data.shape[0],:] = actual_data

		x = torch.tensor(data)
		y = torch.tensor(np.expand_dims(affinity, axis=0))
		#mask = (x[:,0] != 0) & (x[:,1] != 0) & (x[:,2] != 0)
		#print(actual_data.shape, x[mask].shape)

		if self.rmsd_weight == True:
			data_w = 0.5 + self.rmsd_thres - rmsd
			w = torch.tensor(np.expand_dims(data_w, axis=0))
			return x, y, w
		else:
			return x, y

