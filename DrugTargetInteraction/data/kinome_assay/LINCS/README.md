# LINCS HMS KinomeScan data set
## Two assay types available: Kd and percent activity (or percent inhibition; PI)

## File description
Filename | Description
----------|----------
assays | directory for downloaded assay outcomes in csv
list | directory for chemical and protein information
pkd_pi_overlap | directory for assays measured in both Kd and percent inhibition
collect_assays.py | python script to collect assays from csv files

## File description - Assays extracted
Filename | Description
----------|----------
LINCS_kinomescan_kd_inactive_null.tsv | list of inactive chem-kinase pairs with null measurement from Kd assay
LINCS_kinomescan_pi_inactive_null.tsv | list of inactive chem-kinase pairs with null measurement from PI assay
LINCS_kinomescan_kd_nM.tsv | list of chem-kinase pairs with measured Kd values in nM
LINCS_kinomescan_pi_nM.tsv | list of chem-kinase pairs with measured percent activity values, compound concentrations in nM

----------
```
REFERENCES
[1] http://lincs.hms.harvard.edu/db/datasets/
```
