# kilobot-results
Scripts for running, processing and plotting Kilobot simulations and their results.


Directory layout is as follows:

├── kilobox

├── kilobox_build

├── ...

├── **testing**

│   ├── results

│      ├── steadystates

│      ├── test_results

│      ├── trajectories

│      └── video_results

Above, the results directory contains the .csv files in the directory "test_results", and the steadystate and trajectory graphs (PDFs) go in their own respective folders. The "video_results" directory is for the output files of the video processing using OpenCV (kilobot_processing.py).

To run Kilobot simulations for a range of different communication radii use the following command:

```shell
$ python run_ra_simulation.py --[three-valued/boolean-adopt/...] --motion=[MOVING/STATIONARY] (optional:) --vars=[0/10/20/30/40/50/60/70/80/90/100 #these are for the malfunction rates]
```

Running the above for --three-valued and --boolean-adopt, separately, will run simulations for a range of communication radii in [0,20] cm (specified in the run_ra_simulation.py file - RADII list) and then save the output as necessary.

You can then run the following to compare the three-valued and boolean-adopt consensus algorithms:

```shell
$ python3 honeybee_comparisons_ra.py --three-valued --boolean-adopt --motion=moving --sites=2
```
