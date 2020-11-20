README for dataset KINASE


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes (amino acid residues)
m = total number of edges (sequential connections between residues)
N = number of graphs (kinases)

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)
    edges contain binding site triplets based on 3D structure & MSA
    edges do not contain sequential edges as the binding site sequences are not necessarily sequential

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i
    i-th row represents the following amino acid index properties of the i-th amino acid
    row=(hydrophobicity,
        van der Waals volume,
        polarity,
        net charge,
        buried volume,
        accessible surface area in tripeptide,
        accessible surface area in folded protein)

(5)     DS_node_attributes_pssm.txt (n lines)
    matrix of node attributes,
    i-th row represents the position-specific scoring matrix (PSSM) of the i-th amino acid
    kinase->UniRef50, using psiblast, n_iter=3
    values are globally normalized by z-scaling (e.g. (x-globalmean)/globalstdev )
