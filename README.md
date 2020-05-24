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

Selected presentations slides in pdf:

* [31 March 2020](https://gitlab.cern.ch/lciucu/TrackML/-/blob/master/slides/LuizaCiucuTrackML200331.pdf), presented at the ATLAS group at the University of Geneva

* [22 May 2020](https://gitlab.cern.ch/lciucu/TrackML/-/blob/master/slides/LuizaCiucuTrackML200522.pdf), presented to my analysis group, comparing unbalanced and balanced relative to the positive vs negative hits