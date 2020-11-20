# Kinase sequence features

# File description
Filename | Description
----------|----------
kinase_domain_pssm_uniref50 | directory containing kinase domain pssm to uniref50
kinomescan_full_sequence.fasta | full sequence for KinomeScan proteins
kinase_domain_fasta.tar.gz | kinase domain sequence for KinomeScan proteins
kinase_domain_pssm_nr.tar.gz | PSSM for kinase domains against NR database. psiblast, #iter=3
kinase_pssm_nr.tar.gz | PSSM for full kinase sequence against NR database. psiblast, #iter=3
get_pssm_from_fasta.py | python script to run psiblast [1] in parallel to get PSSM
kinases_with_bsite | list of UniProt IDs for kinases with domain PSSM available
extract_domain_and_bsite.py | python script to extract kinase domain and binding site residues. Requires Biopython [2] and interproscan [3]



```
REFERENCES
[1] Altschul, Stephen F., et al. "Gapped BLAST and PSI-BLAST: a new generation of protein database search programs." Nucleic acids research 25.17 (1997): 3389-3402.
[2] Biopython. https://biopython.org/
[3] InterPro. https://www.ebi.ac.uk/interpro/
```
