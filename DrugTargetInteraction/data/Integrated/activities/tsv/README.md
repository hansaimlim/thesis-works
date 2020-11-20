# Integrated chemical-protein activities

### File description -- continuous activity (pKd, pKi, and pIC50)
#### Duplicate entries from multiple databases may be found (not merged)
Filename | Description
----------|----------
integrated_continuous_activity.tsv.gz | all available continuous activities
integrated_pic50.tsv.gz | all available activities in pIC50
integrated_pkd.tsv.gz | all available activities in pKd
integrated_pki.tsv.gz | all available activities in pKi
integrated_tf_continuous.tsv.gz | transcription factors only
integrated_cancer_related_targets_continuous.tsv.gz | cancer-related targets only
integrated_cardio_targets_continuous.tsv.gz | candidate cardiovascular disease targets only
integrated_cyp450_continuous.tsv.gz | cytochrome P450 targets only
integrated_disease_related_targets_continuous.tsv.gz | disease-related targets only
integrated_fda_approved_targets_continuous.tsv.gz | FDA-approved targets only
integrated_gpcr_continuous.tsv.gz | G-protein coupled receptors only
integrated_nr_continuous.tsv.gz | nuclear receptors only
integrated_potential_drug_targets_continuous.tsv.gz | potential drug targets only

---------------------------

### File description -- binary activity (active/inactive)
#### Duplicate entries were merged into one record
Filename | Description
----------|----------
integrated_binary_activity.tsv.gz | all available binary activities
integrated_tf_binary.tsv.gz | transcription factors only
integrated_cancer_related_targets_binary.tsv.gz | cancer-related targets only
integrated_cardio_targets_binary.tsv.gz | candidate cardiovascular disease targets only
integrated_cyp450_binary.tsv.gz | cytochrome P450 targets only
integrated_disease_related_targets_binary.tsv.gz | disease-related targets only
integrated_fda_approved_targets_binary.tsv.gz | FDA-approved targets only
integrated_gpcr_binary.tsv.gz | G-protein coupled receptors only
integrated_nr_binary.tsv.gz | nuclear receptors only
integrated_potential_drug_targets_binary.tsv.gz | potential drug targets only

#### Files were generated using `collect_data.py` script
#### Files are gzipped for compactness


#### Protein target classes were based on:
`https://www.proteinatlas.org/humanproteome/proteinclasses`


#### Protein IDs were mapped using UniProt.
`https://www.uniprot.org/uploadlists/`
