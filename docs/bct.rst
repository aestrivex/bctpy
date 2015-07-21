Brain Connectivity Toolbox
==========================

.. currentmodule::  bct

Centrality
----------

.. autofunction:: betweenness_bin
.. autofunction:: betweenness_wei
.. autofunction:: diversity_coef_sign
.. autofunction:: edge_betweenness_bin
.. autofunction:: edge_betweenness_wei
.. autofunction:: eigenvector_centrality_und
.. autofunction:: flow_coef_bd
.. autofunction:: kcoreness_centrality_bd
.. autofunction:: kcoreness_centrality_bu
.. autofunction:: module_degree_zscore
.. autofunction:: pagerank_centrality
.. autofunction:: participation_coef
.. autofunction:: participation_coef_sign
.. autofunction:: subgraph_centrality

Clustering
----------

.. autofunction:: agreement
.. autofunction:: agreement_weighted
.. autofunction:: clustering_coef_bd
.. autofunction:: clustering_coef_bu
.. autofunction:: clustering_coef_wd
.. autofunction:: clustering_coef_wu
.. autofunction:: consensus_und
.. autofunction:: get_components
.. autofunction:: transitivity_bd
.. autofunction:: transitivity_bu
.. autofunction:: transitivity_wd
.. autofunction:: transitivity_wu

Core
----

.. autofunction:: assortativity_bin
.. autofunction:: assortativity_wei
.. autofunction:: kcore_bd
.. autofunction:: kcore_bu
.. autofunction:: rich_club_bd
.. autofunction:: rich_club_bu
.. autofunction:: rich_club_wd
.. autofunction:: rich_club_wu
.. autofunction:: score_wu

Degree
------

.. autofunction:: degrees_dir
.. autofunction:: degrees_und
.. autofunction:: jdegree
.. autofunction:: strengths_dir
.. autofunction:: strengths_und
.. autofunction:: strengths_und_sign

Distance
--------

.. autofunction:: breadthdist
.. autofunction:: breadth
.. autofunction:: charpath
.. autofunction:: cycprob
.. autofunction:: distance_bin
.. autofunction:: distance_wei
.. autofunction:: efficiency_bin
.. autofunction:: efficiency_wei
.. autofunction:: findpaths
.. autofunction:: findwalks
.. autofunction:: reachdist

Modularity
----------

.. autofunction:: ci2ls
.. autofunction:: ls2ci
.. autofunction:: community_louvain
.. autofunction:: link_communities
.. autofunction:: modularity_dir
.. autofunction:: modularity_und
.. autofunction:: modularity_und_sig
.. autofunction:: modularity_finetune_und
.. autofunction:: modularity_finetune_und_sign
.. autofunction:: modularity_louvain_dir
.. autofunction:: modularity_louvain_und
.. autofunction:: modularity_louvain_und_sign
.. autofunction:: modularity_probtune_und_sign
.. autofunction:: partition_distance

Motif
-----

.. autofunction:: find_motif34
.. autofunction:: motif3funct_bin
.. autofunction:: motif3funct_wei
.. autofunction:: motif3struct_bin
.. autofunction:: motif3struct_wei
.. autofunction:: motif4funct_bin
.. autofunction:: motif4funct_wei
.. autofunction:: motif4struct_bin
.. autofunction:: motif4struct_wei

Miscellaneous
-------------

.. autofunction:: threshold_absolute
.. autofunction:: threshold_proportional 
.. autofunction:: weight_conversion
.. autofunction:: binarize
.. autofunction:: normalize
.. autofunction:: invert
.. autofunction:: autofix

Physical Connectivity
---------------------

.. autofunction:: density_dir
.. autofunction:: density_und
.. autofunction:: rentian_scaling

Reference
---------
.. autofunction:: latmio_dir_connected
.. autofunction:: latmio_dir
.. autofunction:: latmio_und_connected
.. autofunction:: latmio_und
.. autofunction:: makeevenCIJ
.. autofunction:: makefractalCIJ
.. autofunction:: makerandCIJdegreesfixed
.. autofunction:: makerandCIJ_dir
.. autofunction:: makerandCIJ_und
.. autofunction:: makeringlatticeCIJ
.. autofunction:: maketoeplitzCIJ
.. autofunction:: null_model_dir_sign
.. autofunction:: null_model_und_sign
.. autofunction:: randmio_dir_connected
.. autofunction:: randmio_dir
.. autofunction:: randmio_und_connected
.. autofunction:: randmio_und
.. autofunction:: randmio_und_signed
.. autofunction:: randomize_graph_partial_und
.. autofunction:: randomizer_bin_und

Similarity
----------

.. autofunction:: edge_nei_overlap_bd
.. autofunction:: edge_nei_overlap_bu
.. autofunction:: gtom
.. autofunction:: matching_ind
.. autofunction:: matching_ind_und
.. autofunction:: dice_pairwise_und
.. autofunction:: corr_flat_und
.. autofunction:: corr_flat_dir

Visualization
-------------

.. autofunction:: adjacency_plot_und
.. autofunction:: align_matrices
.. autofunction:: backbone_wu
.. autofunction:: grid_communities
.. autofunction:: reorderMAT
.. autofunction:: reorder_matrix
.. autofunction:: reorder_mod
.. autofunction:: writetoPAJ

Network Based Statistic
=======================

.. currentmodule:: nbs

.. autofunction:: nbs_bct
