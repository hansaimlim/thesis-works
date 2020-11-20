# Integrated chemical-protein activities

## File description
Filename | Description
----------|----------
collect_data.py | python script to collect activities
Idmap.py | chemical and protein ID mapping module
utils.py | utility module used in `collect_data.py`
BindingDB.py | BindingDB data module
ChEMBL.py | ChEMBL data module
DrugBank.py | DrugBank data module
GPCRdb.py | GPCRDB data module
KinomeAssay.py | Kinome Assay data module
PubChemCYP450.py | PubChem Cytochrome P450 data module
_matlab_ | directory for matlab-readable matrices for REMAP
_tsv_ | directory for tab-separated, integrated activity tables

-----------

## To obtain chemical-protein activity data by measurement type

* To collect activities in any __continuous activity__ measurements (pKi, pKd, pIC50),
```
python collect_data.py --assay_type continuous
```
* To collect activities in __pKi__ measurement, `python collect_data.py --assay_type pki`
* To collect activities in __pKd__ measurement, `python collect_data.py --assay_type pkd`
* To collect activities __without__ continuous measurement (active/inactive),
```
python collect_data.py --assay_type binary #note that this does NOT contain all activities
```
#### For complete binary data set, you may collect continuous activities and binarize based on your own threshold rules. Note that continuous activities may have inequality measurements (_e.g._ pKd > 6.5). _matlab_ directory contains a version of __complete, binarized__ activities. Refer to the binarizer script in _matlab_ directory for details.

-----------

## To obtain chemical-protein activity data by target protein class
### To be developed...
