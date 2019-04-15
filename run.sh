#!/bin/bash
#python main.py -img test_images.csv -size 100 -step 50 -out ./output/ -bof dict-s25.yml -svm s25_r1_it1_csvm_bestmodel.RData -gt corpus-26000-positivo.csv -ovlp 0.1 &&
#python generate_masks.py -csv ./output/result100_50step.csv &&
#cd cluster &&
#python dbscan.py -csv ../output/result100_50step/binary_masks &&
#cd .. &&
#python validation.py -csv ./output/result100_50step/metrics_cluster_result100_50step.csv
cd postprocessing &&
python cc_processing.py -path ../output/result100_50step/binary_masks -min 2500 -max 10000 -step 2500 &&
python cc_processing.py -path ../output/result200_100step/binary_masks -min 10000 -max 40000 -step 10000 &&
python cc_processing.py -path ../output/result300_150step/binary_masks -min 22500 -max 90000 -step 22500 &&
python cc_processing.py -path ../output/result400_200step/binary_masks -min 40000 -max 160000 -step 40000 &&
python cc_processing.py -path ../output/result500_250step/binary_masks -min 62500 -max 250000 -step 62500 &&
python cc_processing.py -path ../output/result600_300step/binary_masks -min 90000 -max 360000 -step 90000 &&
python cc_processing.py -path ../output/result700_350step/binary_masks -min 122500 -max 490000 -step 122500 


