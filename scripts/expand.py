import openeye
import openforcefield
from tqdm import tqdm

print(openforcefield.__version__)
print(openeye.__version__)

from openforcefield.topology import Molecule

path = '../data/Enamine_Discovery_Diversity_Set_plated_10240cmds_20200806.sdf'

mols = Molecule.from_file(path, file_format='sdf', allow_undefined_stereo=True)

# for each molecule, enumerate protomers and tautomers
expanded_set = []

for mol in tqdm(mols):
    expanded_set.append(mol)
    expanded_set.extend(mol.enumerate_protomers())
    expanded_set.extend(mol.enumerate_tautomers())

# save as list of isomeric SMILES
expanded_smiles = [mol.to_smiles() for mol in expanded_set]

print('originals', len(mols))
print('expanded', sum(list(map(len, expanded_set))))

# check unique
unique = set(expanded_smiles)
print('unique', len(unique))

n_unique = len(unique)
in_thousands = int(round(n_unique / 1000))

with open(f'../data/Enamine_Discovery_10K_expanded_{in_thousands}K.smi', 'w') as f:
    f.writelines(list(unique))
