# SlidingWindowsBudDetection
Run main.py to generate result_csv

Setup the base image path in generate_masks.py (original large scale images folder: IMAGES_PATH) and run with result_csv

Then comes clustering, run dbscan.py with cluster_route updated to the folder where bin masks are:
* eg: .../SlidingWindowsBudDetection/output/result300_150step/binary_masks,
* Then you get a metrics_cluster_... .csv
   
Then you need to generate a validation plot so run validation.py on base dir.

Pass the path to metrics_cluster_... .csv generated by dbscan.py

