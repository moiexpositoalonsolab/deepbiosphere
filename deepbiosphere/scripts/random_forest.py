import pandas as pd
from deepbiosphere.scripts.GEOCLEF_Config import paths, Run_Params
import deepbiosphere.scripts.GEOCLEF_Config as config
# Import train_test_split function
from sklearn.model_selection import train_test_split
from deepbiosphere.scripts import GEOCLEF_Dataset as dataset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def main(ARGS):
    print("loading dataset")
    dset = dataset.Bioclim_Rasters_Point(ARGS.base_dir, 'plant', 'cali', 'min_max', 'joint_multiple')
    # base_dir, organism, region, normalize, observation):
    # obs[['id', 'species_id', 'genus_id', 'family_id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon']].values
    occs = pd.read_csv('{}occurrences/joint_multiple_obs_cali_plant_train.csv'.format(ARGS.base_dir))
    most_freq = occs.species.value_counts().keys()[:10]
    most_freq_ids = [dset.spec_dict[m] for m in most_freq]
    obs = dset.obs
    rasters = dset.rasters
    affine = dset.affine
    lenlen=0
    for i, ob in enumerate(obs):
        sp_id = ob[1]
        if sp_id in most_freq_ids:
            lenlen += 1
    rebuilt = np.zeros([22, lenlen])
    # print(rebuilt.shape)
    tracker = 0
    for i, ob in enumerate(obs):
        lat_lon = ob[7]
        sp_id = ob[1]
        if sp_id in most_freq_ids:
            env_rasters = dataset.get_raster_point_obs(lat_lon, dset.affine, dset.rasters, dset.nan, dset.normalize, dset.lat_min, dset.lat_max, dset.lon_min, dset.lon_max)
            rebuilt[0, tracker] = sp_id
            ind = 1
	
            for ras in env_rasters:
        #         print(len(env_rasters))
                rebuilt[ind, tracker] = ras
                ind+=1
            tracker += 1
    dframe = pd.DataFrame(rebuilt.T,index=np.arange(lenlen), columns=['spec_id', 'bio_1', 'bio_2','bio_3','bio_4','bio_5', 'bio_6', 'bio_7','bio_8','bio_9','bio_10', 'bio_11', 'bio_12','bio_13','bio_14','bio_15','bio_16', 'bio_17','bio_18','bio_19','lat', 'lon'])

    # X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
    X=dframe.loc[:, dframe.columns != 'spec_id']# dframe[~'spec_id']  # Labels
    y=dframe['spec_id']  # Labels
    print("splitting data")
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # 70% training and 30% test

    print("training classaifier")

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=50)
    print("let's go")
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    print("predicintg uising classifier")
    y_pred=clf.predict(X_test)
    #Import scikit-learn metrics module for accuracy calculation

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    # add CLI to nuke a whole model
    # add CLI to clean up all model files



    args = [ 'base_dir']
    ARGS = config.parse_known_args(args)
    main(ARGS)
