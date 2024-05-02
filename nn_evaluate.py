#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Autoencoders evaluation.

Usage:
  nn_evaluate.py [--whole] [--male] [--threshold] [--leave-site-out] [--NYU-site-out] [<derivative> ...]
  nn_evaluate.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""
import numpy as np
import pandas as pd
#import tensorflow as tf
from docopt import docopt
from nn import nn
from utils import (load_phenotypes, format_config, hdf5_handler,
                   reset, to_softmax, load_ae_encoder, load_fold)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

import tensorflow.compat.v1 as tf
from sklearn.feature_selection import SelectKBest,f_classif


def nn_results(hdf5, experiment, code_size_1, code_size_2):

    exp_storage = hdf5["experiments"][experiment]

    n_classes = 2

    results = []

    count=0
    for fold in exp_storage:
        if count==1:
          break
        count+=1
        experiment_cv = format_config("{experiment}_{fold}", {
            "experiment": experiment,
            "fold": fold,
        })

        
        X_train, y_train, \
        X_valid, y_valid, \
        X_test, y_test = load_fold(hdf5["patients"], exp_storage, fold)


        X_all=np.vstack((X_train,X_valid,X_test))
        y_all=np.concatenate((np.array(y_train),np.array(y_valid),np.array(y_test)),axis=0)

        # print(X_all.shape)
        # print(y_all.shape)

        ks=0
        if X_train.shape[1]<10000:
          ks=1000
        else:
          ks=3000    
        X_new=SelectKBest(f_classif,k=ks).fit_transform(X_all[:,:-2], y_all)
        # print(X_new.shape)

        train=X_train.shape[0]
        valid=X_valid.shape[0]
        test=X_test.shape[0]

        X_new=np.concatenate((X_new,X_all[:,-2:]),axis=1)

        X_train=X_new[:train]
        X_valid=X_new[train:train+valid]
        X_test=X_new[train+valid:train+valid+test]

        # print(X_new.shape)

        #y_test = np.array([to_softmax(n_classes, y) for y in y_test])


        #X_test=X_new[:]
        y_test = np.array([to_softmax(n_classes, y) for y in y_all])

        '''
        X_NYU=np.load('X_NYU.npy')
        y_NYU=np.load('y_NYU.npy')

        X_sub=np.load('/content/drive/MyDrive/acerta-abide/X_sub_without_NYU.npy')
        y_sub=np.load('/content/drive/MyDrive/acerta-abide/y_sub_without_NYU.npy')
        
        X=np.vstack((X_NYU,X_sub))
        y=np.concatenate((y_NYU,y_sub))

        ks=0
        if X.shape[1]<10000:
          ks=1000
        else:
          ks=3000
        X_new=SelectKBest(f_classif,k=ks).fit_transform(X, y)
        print(X_new.shape)

        
        X_train,X_test,y_train,y_test = train_test_split(X_new,y,test_size=0.1,random_state=0)
        X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.33,random_state=0)
        '''


        ae1_model_path = format_config("./data/models/{experiment}_autoencoder-1.ckpt", {
            "experiment": experiment_cv,
        })
        ae2_model_path = format_config("./data/models/{experiment}_autoencoder-2.ckpt", {
            "experiment": experiment_cv,
        })
        nn_model_path = format_config("./data/models/{experiment}_mlp.ckpt", {
            "experiment": experiment_cv,
        })

        try:

            model = nn(X_test.shape[1], n_classes, [
                {"size": 1000, "actv": tf.nn.tanh},
                {"size": 600, "actv": tf.nn.tanh},
                {"size": 100, "actv": tf.nn.tanh},

            ])

            init = tf.compat.v1.global_variables_initializer()
            #init = tf.global_variables_initializer()
            with tf.Session() as sess:

                sess.run(init)

                saver = tf.train.Saver(model["params"])
                saver.restore(sess, nn_model_path)

                output = sess.run(
                    model["output"],
                    feed_dict={
                        model["input"]: X_new,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0,
                        model["dropouts"][2]: 1.0,
                    }
                )

                y_pred = np.argmax(output, axis=1)
                y_true = np.argmax(y_test, axis=1)

                # print(y_true.shape)
                # print(y_pred.shape)
                # print(y_pred)
                # print(y_true)

                # print(sum(np.equal(y_pred,y_true)))

                X_sub=[]
                y_sub=[]

                for i in range(X_new.shape[0]):
                  if y_pred[i]==y_true[i]:
                    X_sub.append(X_all[i])
                    y_sub.append(y_pred[i])

                X_sub=np.array(X_sub)
                # print(X_sub.shape)
                y_sub=np.array(y_sub)

                # print(X_sub.shape)

                np.save('X_sub_without_NYU.npy',X_sub)
                np.save('y_sub_without_NYU.npy',y_sub)

                [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
                specificity = TN/(FP+TN)
                precision = TP/(TP+FP)
                sensivity = TP/(TP+FN)

                accuracy = accuracy_score(y_true, y_pred)
                fscore = f1_score(y_true, y_pred)
                roc_auc = roc_auc_score(y_true, y_pred)

                results.append([accuracy, precision, fscore, sensivity, specificity, roc_auc])
        finally:
            reset()

    return [experiment] + np.mean(results, axis=0).tolist()

if __name__ == "__main__":

    reset()

    arguments = docopt(__doc__)

    pd.set_option("display.expand_frame_repr", False)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    hdf5 = hdf5_handler(bytes("./data/abide.hdf5",encoding="utf8"), "a")

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative
                   in arguments["<derivative>"]
                   if derivative in valid_derivatives]

    experiments = []

    for derivative in derivatives:

        config = {"derivative": derivative}

        if arguments["--whole"]:
            experiments += [format_config("{derivative}_whole", config)],

        if arguments["--male"]:
            experiments += [format_config("{derivative}_male", config)]

        if arguments["--threshold"]:
            experiments += [format_config("{derivative}_threshold", config)]

        if arguments["--leave-site-out"]:
            for site in pheno["SITE_ID"].unique():
                if site=='NYU':
                  site_config = {"site": site}
                  experiments += [
                      format_config("{derivative}_leavesiteout-{site}",
                                    config, site_config)
                  ]

        if arguments["--NYU-site-out"]:
            experiments += [format_config("{derivative}_leavesiteout-NYU", config)]


    # First autoencoder bottleneck
    code_size_1 = 1000 # not used

    # Second autoencoder bottleneck
    code_size_2 = 600 # not used

    results = []

    experiments = sorted(experiments)
    print(experiments)
    for experiment in experiments:
        print(experiment)
        results.append(nn_results(hdf5, experiment, code_size_1, code_size_2))

    cols = ["Exp", "Accuracy", "Precision", "F1-score", "Sensivity", "Specificity", "ROC-AUC"]
    df = pd.DataFrame(results, columns=cols)

    print('aaa',df[cols] \
        .sort_values(["Exp"]) \
        .reset_index())
