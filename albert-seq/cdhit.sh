#!/bin/bash
for filename in all_pfam_fasta/*; do
	f=${filename##*/}
	cd-hit  -i all_pfam_fasta/${f} \
		-o pfam_cdhit_clusters90/${f} -c 0.9 -n 5 \
		-g 1 -G 0 -aS 0.8 \
		-d 0 -p 1 -T 16 -M 0 > pfam_cdhit_cluster_log/${f}.log
	cd-hit  -i pfam_cdhit_clusters90/${f} \
		-o pfam_cdhit_clusters60/${f} -c 0.6 -n 4 \
		-g 1 -G 0 -aS 0.8 \
		-d 0 -p 1 -T 16 -M 0 > pfam_cdhit_cluster_log/${f}.log

	psi-cd-hit.pl -i pfam_cdhit_clusters60/${f} -o pfam_cdhit_clusters30/${f} \
		-c 0.3 -ce 1e-6 -aS 0.8 -G 0 -g 1 -exec local -para 8 -blp 4
	clstr_rev.pl pfam_cdhit_clusters90/${f}.clstr pfam_cdhit_clusters60/${f}.clstr > pfam_cdhit_clusters9060/${f}.clstr
	clstr_rev.pl pfam_cdhit_clusters9060/${f}.clstr pfam_cdhit_clusters30/${f}.clstr > pfam_cdhit_clusters906030/${f}.clstr
done
