# GalaxyClassifier

1. Create the following directories if they don't already exist:
   1. data/
   2. models/
2. In addition, go to find the original data used https://www.kaggle.com/datasets/jaimetrickz/galaxy-zoo-2-images
3. Unzip those images to the data/images/ directory and run the data_pipeline.split_datasets() function. This should sort the images into the train/test/validate folders.
   1. **note:** if those 3 directories are not created, you can manually create them in the data/ directory (just 3 folders called train/test/validate respectively). These directories will need to be empty and run only once.