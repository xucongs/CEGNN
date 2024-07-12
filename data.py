import os
import csv
import json
import math
import torch
import numpy as np
import warnings
from torch.utils.data import Dataset
from pymatgen.core.structure import Structure

class AtomInitializer(object):
    """
    Base class for initializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2, max_node_num=100):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius, self.max_node_num = max_num_nbr, radius, max_node_num
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'classfi_tot.csv')
        
        assert os.path.exists(id_prop_file), 'classfi.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        
        cif_id1, cif_id2, target = self.id_prop_data[idx]
        crystal1 = Structure.from_file(os.path.join(self.root_dir, cif_id1 + '.cif'))
        crystal2 = Structure.from_file(os.path.join(self.root_dir, cif_id2 + '.cif'))

        atom_fea1, atom_coords1, nbr_fea_idx1, nbr_fea1, edge_mask1, atom_mask1 = self._process_crystal(crystal1)
        atom_fea2, atom_coords2, nbr_fea_idx2, nbr_fea2, edge_mask2, atom_mask2 = self._process_crystal(crystal2)

        atom_fea1, atom_coords1, nbr_fea_idx1, nbr_fea1, edge_mask1, atom_mask1 = self._pad_features(
            atom_fea1, atom_coords1, nbr_fea_idx1, nbr_fea1, edge_mask1, atom_mask1)
        atom_fea2, atom_coords2, nbr_fea_idx2, nbr_fea2, edge_mask2, atom_mask2 = self._pad_features(
            atom_fea2, atom_coords2, nbr_fea_idx2, nbr_fea2, edge_mask2, atom_mask2)

        target = torch.Tensor([float(target)])

        lattice1 = crystal1.lattice
        lattice_matrix1 = construct_lattice_matrix(lattice1.a, lattice1.b, lattice1.c, lattice1.alpha, lattice1.beta, lattice1.gamma)

        lattice2 = crystal2.lattice
        lattice_matrix2 = construct_lattice_matrix(lattice2.a, lattice2.b, lattice2.c, lattice2.alpha, lattice2.beta, lattice2.gamma)

        return (atom_fea1, atom_coords1, lattice_matrix1, edge_mask1, atom_mask1), \
               (atom_fea2, atom_coords2, lattice_matrix2, edge_mask2, atom_mask2), \
               target

    def _process_crystal(self, crystal):
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].species.elements[0].number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)

        atom_coords = np.vstack([crystal[i].coords for i in range(len(crystal))])
        atom_coords = torch.Tensor(atom_coords)

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        node_num = len(atom_fea)
        nbr_fea_idx, nbr_fea, edge_mask, atom_mask = self._get_neighbor_features_and_mask(all_nbrs, node_num)

        return atom_fea, atom_coords, nbr_fea_idx, nbr_fea, edge_mask, atom_mask

    def _get_neighbor_features_and_mask(self, all_nbrs, node_num):
        nbr_fea_idx, nbr_fea = [], []
        edge_mask = np.zeros((self.max_node_num, self.max_node_num), dtype=np.float32)
        atom_mask = np.zeros(self.max_node_num, dtype=np.float32)

        for i, nbr in enumerate(all_nbrs):
            if i >= self.max_node_num:
                break
            if len(nbr) < self.max_num_nbr:
                warnings.warn('Not enough neighbors to build graph.')
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
                for n in nbr:
                    if n[2] < self.max_node_num:
                        edge_mask[i, n[2]] = 1
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
                for n in nbr[:self.max_num_nbr]:
                    if n[2] < self.max_node_num:
                        edge_mask[i, n[2]] = 1
            atom_mask[i] = 1

        return nbr_fea_idx, nbr_fea, edge_mask, atom_mask

    def _pad_features(self, atom_fea, atom_coords, nbr_fea_idx, nbr_fea, edge_mask, atom_mask):
        node_num = len(atom_fea)
        if node_num < self.max_node_num:
            padding = self.max_node_num - node_num
            atom_fea = torch.cat([atom_fea, torch.zeros((padding, atom_fea.shape[1]))], dim=0)
            atom_coords = torch.cat([atom_coords, torch.zeros((padding, atom_coords.shape[1]))], dim=0)
            nbr_fea_idx = np.array(nbr_fea_idx + [[0] * self.max_num_nbr] * padding)
            nbr_fea = np.array(nbr_fea + [[self.radius + 1.] * self.max_num_nbr] * padding)
        else:
            atom_fea = atom_fea[:self.max_node_num]
            atom_coords = atom_coords[:self.max_node_num]
            nbr_fea_idx = np.array(nbr_fea_idx[:self.max_node_num])
            nbr_fea = np.array(nbr_fea[:self.max_node_num])

        atom_fea = torch.Tensor(atom_fea)
        atom_coords = torch.Tensor(atom_coords)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        edge_mask = torch.Tensor(edge_mask)
        atom_mask = torch.Tensor(atom_mask)

        return atom_fea, atom_coords, nbr_fea_idx, nbr_fea, edge_mask, atom_mask


def construct_lattice_matrix(a, b, c, alpha, beta, gamma):
    """
    Construct the lattice matrix from lattice parameters.
    
    Args:
    - a, b, c (float): Lattice constants.
    - alpha, beta, gamma (float): Lattice angles in degrees.
    
    Returns:
    - lattice_matrix (torch.Tensor): Lattice matrix of shape (3, 3)
    """
    alpha_r = math.radians(alpha)
    beta_r = math.radians(beta)
    gamma_r = math.radians(gamma)
    
    v = math.sqrt(1 - math.cos(alpha_r) ** 2 - math.cos(beta_r) ** 2 - math.cos(gamma_r) ** 2 +
                  2 * math.cos(alpha_r) * math.cos(beta_r) * math.cos(gamma_r))
    
    lattice_matrix = torch.tensor([
        [a, b * math.cos(gamma_r), c * math.cos(beta_r)],
        [0, b * math.sin(gamma_r), c * (math.cos(alpha_r) - math.cos(beta_r) * math.cos(gamma_r)) / math.sin(gamma_r)],
        [0, 0, c * v / math.sin(gamma_r)]
    ])
    
    return lattice_matrix

def collate_fn(batch):
    data1, data2, target = zip(*batch)
    
    atom_fea1, atom_coords1, lattice_matrix1, edge_mask1, atom_mask1 = zip(*data1)
    atom_fea2, atom_coords2, lattice_matrix2, edge_mask2, atom_mask2 = zip(*data2)
    
    atom_fea1 = torch.stack(atom_fea1, dim=0)
    atom_coords1 = torch.stack(atom_coords1, dim=0)
    lattice_matrix1 = torch.stack(lattice_matrix1, dim=0)
    edge_mask1 = torch.stack(edge_mask1, dim=0)
    atom_mask1 = torch.stack(atom_mask1, dim=0)
    
    atom_fea2 = torch.stack(atom_fea2, dim=0)
    atom_coords2 = torch.stack(atom_coords2, dim=0)
    lattice_matrix2 = torch.stack(lattice_matrix2, dim=0)
    edge_mask2 = torch.stack(edge_mask2, dim=0)
    atom_mask2 = torch.stack(atom_mask2, dim=0)
    
    target = torch.stack(target, dim=0)
    
    return (atom_fea1, atom_coords1, lattice_matrix1, edge_mask1, atom_mask1), \
           (atom_fea2, atom_coords2, lattice_matrix2, edge_mask2, atom_mask2), \
           target


def preprocess_and_save(dataset, save_path):
    data_list = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        data_list.append(data)
    torch.save(data_list, save_path)
    
class PreprocessedCIFData(Dataset):
    def __init__(self, data_path):
        self.data_list = torch.load(data_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
