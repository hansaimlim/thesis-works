from rdkit import Chem
from features import *
from graph import node_id, Graph, Molecule, load_from_smiles

degrees=[0,1,2,3,4,5] #make sure it is the same as in fingerprint.graph.py

def get_molecule_dict(chemfile):
    """ preprocess molecules
        returns dict of molecules
        key: InChIKey
        values: dict(bond-bond_features), 
               dict(atom_features),
               dict(neighbor_idx_by_degree)
    """
    molecule_dict={}
    with open(chemfile,'r') as f:
        for line in f:
            line=line.strip().split('\t')
            ikey=line[0]
            smi=line[1]
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                raise ValueError("Could not generate Mol from SMILES string:", smi)
            #Chem.SanitizeMol(mol)

            atoms={} #atom_idx -> atom features
            bonds={} #bond_idx -> bond features
            atoms2bond={} #(atom_idx1,atom_idx2) -> bond_idx
            
            nodes_by_degree = {d: [] for d in degrees}
            for atom in mol.GetAtoms():
                atom_feature = atom_features(atom)
                atom_id = smi+str(atom.GetIdx())
                atoms[atom.GetIdx()]=atom_feature 
            atom_neighbors={aid: [] for aid in atoms.keys()} #atom_idx -> neighbor atom idxs
            bond_neighbors={aid: [] for aid in atoms.keys()} #atom_idx -> neighbor bond idxs

            for bond in mol.GetBonds():
                src_atom_idx = bond.GetBeginAtom().GetIdx()
                tgt_atom_idx = bond.GetEndAtom().GetIdx()
                bond_idx = bond.GetIdx()
                bond_neighbors[src_atom_idx].append(bond_idx)
                bond_neighbors[tgt_atom_idx].append(bond_idx)
                bond_feature = bond_features(bond)
                bonds[bond.GetIdx()] = bond_feature
                atom_neighbors[src_atom_idx].append(tgt_atom_idx)
                atom_neighbors[tgt_atom_idx].append(src_atom_idx)
                atoms2bond[(src_atom_idx,tgt_atom_idx)]=bond_idx
                atoms2bond[(tgt_atom_idx,src_atom_idx)]=bond_idx
            
            atoms_by_degree={d: [] for d in degrees}
            bonds_by_degree={d: [] for d in degrees}
            for aid in atom_neighbors:
                neighbor_atoms = atom_neighbors[aid]
                d = len(neighbor_atoms) #degree of the atom
                atoms_by_degree[d].append(aid) #current atom is degree=d
                neighbor_bonds=[]
                for neighbor in neighbor_atoms:
                    bond_idx=atoms2bond[(aid,neighbor)]
                    neighbor_bonds.append(bond_idx)
                bonds_by_degree[d].append(neighbor_bonds)

            neighbor_by_degree = []
            for degree in degrees:
                neighbor_by_degree.append({
                    'atom': atoms_by_degree[degree],
                    'bond': bonds_by_degree[degree]
                })
            
            molecule_dict[ikey]={'smiles':str(smi),
            'neighbor_by_degree':neighbor_by_degree,
            'atoms':atoms,'bonds':bonds,
            'atom_neighbor':atom_neighbors,
            'bond_neighbor':bond_neighbors}
    return molecule_dict

if __name__=='__main__':
    moldict=get_molecule_dict('./test_chem.tsv')
    print(len(moldict))
    print(moldict['AAGVCJFRSHQFKK-CMDGGOBGSA-N'])
    print(moldict['NIJJYAXOARWZEE-UHFFFAOYSA-N'])
    print(moldict['ODPKTGAWWHZBOY-UHFFFAOYSA-N'])

