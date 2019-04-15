#!/bin/bash
#python main.py -img test_images.csv -size 100 -step 50 -out ./output/ -bof dict-s25.yml -svm s25_r1_it1_csvm_bestmodel.RData -gt corpus-26000-positivo.csv -ovlp 0.1 &&
#python generate_masks.py -csv ./output/result100_50step.csv &&
#cd cluster &&
#python dbscan.py -csv ../output/result100_50step/binary_masks &&
#cd .. &&
#python validation.py -csv ./output/result100_50step/metrics_cluster_result100_50step.csv
cd postprocessing &&
python cc_processing.py -path ../output/result400_200step/binary_masks -min 40000 -max 160000 -step 40000 &&
cd .. &&
python validation.py -csv ./output/result400_200step/th40000_metrics_cluster_result400_200step.csv &&
python validation.py -csv ./output/result400_200step/th80000_metrics_cluster_result400_200step.csv &&
python validation.py -csv ./output/result400_200step/th120000_metrics_cluster_result400_200step.csv &&
python validation.py -csv ./output/result400_200step/th160000_metrics_cluster_result400_200step.csv

