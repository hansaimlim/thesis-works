# Integrated information for proteins

## Protein list and ID coversion script available

## File description
Filename | Description
----------|----------
mutant_id_conversion.tsv | List of mutant proteins 
search_protein_id.py | Python script to convert gene names to uniprot IDs
uniprot_id_mapping.tsv | List of wild-type proteins, may include isoforms
cytochrome_p450_id_mapping.tsv | List of cytochrome P450 proteins. Canonical only.
nuclear_receptor_id_mapping.tsv | List of nuclear receptor proteins. Canonical only.
transcription_factor_id_mapping.tsv | List of transcription factor proteins. Canonical only.
prot_prot_sim_blast_1e8.tsv.gz | gzip compressed list of __sequence-based protein-protein similarity__
pssm_uniref50 | directory for __PSSM__ against __UniRef50__ database
blastdb | directory for local BLAST DB to calculate sequence-based protein-protein similarity

## File description details

Filename | Description
----------|----------
prot_prot_sim_blast_1e8.tsv.gz | __e-value=1e-8__, and default options for BLASTP. Self-similarity (identity) __excluded__. May be asymmetric.
pssm_uniref50 | __n_iter=3__, and default options for psiblast. Target DB is __UniRef__ sequences clustered by __50% identity__.
blastdb | `makeblastdb -in whole_fasta.fas -dbtype prot -parse_seqids -out integrated_proteins` __Commas__ in protein IDs are automatically __converted to '_'__

-------------
* BLAST Standalone version 2.8.1 was used.
