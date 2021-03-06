The master thesis project in experimental particle physics at the University of Geneva.

Implementing the project of matching hits to a particle from this [Kaggle competition](https://www.kaggle.com/c/trackml-particle-identification) in a `Jupyter Notebook` with `Python3`, `Keras`, `TensorFlow`, and plotting in `matplotlib`.

It contains four main steps:

* `DataExploration.ipynb`

* `Data.ipynb` (create input and output for NN training, with and without balancing the positive and negative hits)

* `NN.ipynb` (train NN, predict from NN, evaluate the predicted model, including also for each `VolumeID`)

* `Overlay.ipynb` (create plots that overlay several models in several ways)

There are also other scripts from various studies:

* `DataExplorationCheckForCosmicRays.ipynb` (checked for truth particles what is the closest point to the beam axis, confirming that all collisions are simulated in the center of the detector, and there are no cosmic rays simulated)

* `ConfusionMatrix.ipynb` (computes the confusion matrix from the numpy arrays of true positive, false positive, false negative, true negative)

* `OverlaySeveralBuckets.ipynb` (plots that overlays several buckets)

Master thesis report and presentation

* [master thesis report](./thesis/LuizaCiucuMScThesisTrackML.pdf)

* [master thesis presentation](./slides/LuizaCiucuTrackML210308.pdf).

Selected presentations slides in pdf:

* [31 March 2020](./slides/LuizaCiucuTrackML200331.pdf), presented at the ATLAS group at the University of Geneva

* [22 May 2020](./slides/LuizaCiucuTrackML200522.pdf), presented to my analysis group, comparing unbalanced and balanced relative to the positive vs negative hits

* [19 Jun 2020](./slides/LuizaCiucuTrackML200619.pdf), presented to my analysis group, comparing three methods of balancing the datasets

* [26 Jun 2020](./slides/LuizaCiucuTrackML200626.pdf), presented to my analysis group, further balancing on all events

* [03 Jul 2020](./slides/LuizaCiucuTrackML200703.pdf), presented to my analysis group, refine balancing in train sample, test sample is unbalanced, hyper-parameter tuning

* [10 Jul 2020](./slides/LuizaCiucuTrackML200710.pdf), presented to my analysis group, added dropout layer

* [17 Jul 2020](./slides/LuizaCiucuTrackML200717.pdf), presented to my analysis group, final results, after added particle reconstruction efficiency