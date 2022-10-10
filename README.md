# hiring-challenge

The project uses python3.10 (important for some `typing` syntax). 

* `checkpoints/` : A checkpoint for the `fight_classifier.model.image_based_model.ProjFromFeatures` model. The notebook `2. Image-based classifiers.ipynb` shows how to load it.
* `dataset/` : The dataset where I load the data. I used [DVC](https://dvc.org/doc/start/data-management/data-versioning) to synchronize across computers, but this requires access to my personal s3 account.
* `fight_classifier/` : Contains the python package with all of the code, as well as the notebooks used to describe the work.
* `joblib_random_forest`: I tried to save a naive model (the random forest of notebook `1. Images augmentation and Clever-Hans effect.ipynb`) with joblib. I could not do the same with the interesting model. I recommend to install the `fight_classifier` module and run the notebook `2. Image-based classifiers.ipynb` in order to run the model.
