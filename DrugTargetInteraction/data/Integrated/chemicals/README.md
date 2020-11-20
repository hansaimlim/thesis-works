# Directory for chemical-chemical similarity

## Environment
* python=3.7.2
* json library - to utilize pre-calculated chem-chem similarity information
* urllib3 module - search through PubChem PUGREST [1] and NIH CACTUS [2] services
* RDKit [3] - convert chemical IDs
* ChemAxon MadFast [4] - calculate chem-chem similarity
## File description
Filename | Description
----------|---------
integrated_InChIKey2Index.json.gz | json file for dict{InChIKey : Index}
integrated_Index2InChIKeys.json.gz | json file for dict{Index : list_of_InChIKeys}
integrated_Index2SMILES.json.gz | json file for dict{Index : SMILES}
integrated_chemicals.tsv.gz | list of Index\tInChIKey\tSMILES
integrated_chemicals.smi.gz | list of SMILES\tIndex (input for MadFast)
get_tani_sim_madfast_all2all.py | python script to calculate chemical-chemical Tanimoto similarity using ChemAxon MadFast
python_code_blocks.py | python code blocks used to preprocess chemical information
process_new_chemicals.py | python code to __collect__ _new chemicals_, __calculate similarity__, and update the information lists
integrated_chem_chem_sim_threshold03_reduced.csv.gz | comma-delimited chem-chem Tanimoto similarity file, __reduced file size__ (_details below_)


## How to use chemical-chemical similarity information

1. Decompress __integrated_chem_chem_sim_threshold03_reduced.csv.gz__
2. Decompress __integrated_InChIKey2Index.json.gz__, __integrated_Index2InChIKeys.json.gz__ and __integrated_Index2SMILES.json.gz__
3. Each chemical is represented by integer index appearing in __integrated_Index2InChIKeys.json.gz__ and __integrated_Index2SMILES.json.gz__
4. _Each line_ represents a pair of similar chemical, delimited by commas. __(Index1,Index2,Similarity)__
5. For example, __46001,62923,0.674419__ represents similarity score of __0.674419__ between __chemical#46001 and chemical#62923__
6. Chemical#46001 (Fc1ccc(cc1Cl)C(=O)N2c3ccccc3Sc4ccccc24) <-> Chemical#62923 (COc1ccc(cc1F)C(=O)N2c3ccccc3Sc4ccccc24) can be found in __integrated_Index2SMILES.json__
7. Similarity information __is symmetric__. 46001,62923,0.674419 also represents 62923,46001,0.674419.
8. Lower-triangle of similarity matrix is __excluded__ to reduce file size. (excluded if Index1 > Index2 as they are symmetric and redundant)
9. Self-similarities are always 1.00 and __excluded__ to reduce file size. (46001,46001,1.00 is excluded)


## How to update with new chemicals
1. Prepare a list of new chemicals in __"InChIKey\tSMILES"__ format.
2. Edit the __new_chemical_file__ variable in __process_new_chemicals.py__
3. For example, __new_chemical_file='./my_new_chemicals.tsv'__ in the __if __name__=='__main__':__ block of the script.
4. Run `python process_new_chemicals.py`
5. Updated file names will be displayed.
6. Update github repository if necessary.
* __MadFast is required__
* Note that JSON keys must be strings, not integers. It may cause conflicts for Index2SMILES or Index2InChIKeys.

-----------


```
REFERENCES
[1] PubChem PUGREST. https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
[2] NIH CACTUS. https://cactus.nci.nih.gov/blog/?p=456
[3] RDKit. https://www.rdkit.org/
[4] ChemAxon MadFast. https://chemaxon.com/products/madfast
```
