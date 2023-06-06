#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Autoencoders training and fine-tuning.

Usage:
  nn.py [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]
  nn.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""

import os
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest,chi2,f_classif


from docopt import docopt
from utils import (load_phenotypes, format_config, hdf5_handler, load_fold,
                   sparsity_penalty, reset, to_softmax, load_ae_encoder)

from model import ae, nn

from sklearn.decomposition import PCA  
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split


def run_autoencoder1(experiment,
                     X_train, y_train, X_valid, y_valid, X_test, y_test,
                     model_path, code_size=1000):
    """

    Run the first autoencoder.
    It takes the original data dimensionality and compresses it into `code_size`

    """

    # Hyperparameters
    learning_rate = 0.0001
    sparse = True  # Add sparsity penalty
    sparse_p = 0.2
    sparse_coeff = 0.5
    corruption = 0.7  # Data corruption ratio for denoising
    ae_enc = tf.nn.tanh  # Tangent hyperbolic
    ae_dec = None  # Linear activation

    training_iters = 700
    batch_size = 100
    n_classes = 2

    if os.path.isfile(model_path) or \
       os.path.isfile(model_path + ".meta"):
        return
    
    tf.disable_v2_behavior()
    print(len(X_train[100]))
    # Create model and add sparsity penalty (if requested)
    model = ae(X_train.shape[1], code_size, corruption=corruption, enc=ae_enc, dec=ae_dec)
    print(X_train.shape[1])
    if sparse:
        model["cost"] += sparsity_penalty(model["encode"], sparse_p, sparse_coeff)

    # Use GD for optimization of model cost
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model["cost"])

    # Initialize Tensorflow session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Define model saver
        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)

        # Initialize with an absurd cost for model selection
        prev_costs = np.array([9999999999] * 3)

        for epoch in range(training_iters):

            # Break training set into batches
            batches = range(int(len(X_train) / batch_size))
            costs = np.zeros((len(batches), 3))

            for ib in batches:

                # Compute start and end of batch from training set data array
                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                # Select current batch
                batch_xs, batch_ys = X_train[from_i:to_i], y_train[from_i:to_i]

                # Run optimization and retrieve training cost
                _, cost_train = sess.run(
                    [optimizer, model["cost"]],
                    feed_dict={
                        model["input"]: batch_xs
                    }
                )

                # Compute validation cost
                cost_valid = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: X_valid
                    }
                )

                # Compute test cost
                cost_test = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: X_test
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]

            # Compute the average costs from all batches
            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs

            # Pretty print training info

            print (
                "Exp={experiment}, Model= ae1, Iter={epoch:5d}, Cost={cost_train:.6f} {cost_valid:.6f} {cost_test:.6f}",
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "cost_train": cost_train,
                    "cost_valid": cost_valid,
                    "cost_test": cost_test,
                } 
                )

        

            # Save better model if optimization achieves a lower cost
            if cost_valid < prev_costs[1]:
                print ("Saving better model")
                saver.save(sess, model_path)
                prev_costs = costs
            else:
                print


def run_autoencoder2(experiment,
                     X_train, y_train, X_valid, y_valid, X_test, y_test,
                     model_path, prev_model_path,
                     code_size=600, prev_code_size=1000):
    """

    Run the second autoencoder.
    It takes the dimensionality from first autoencoder and compresses it into the new `code_size`
    Firstly, we need to convert original data to the new projection from autoencoder 1.

    """

    if os.path.isfile(model_path) or \
       os.path.isfile(model_path + ".meta"):
        return

    tf.disable_v2_behavior()

    # Convert training, validation and test set to the new representation
    prev_model = ae(X_train.shape[1], prev_code_size,
                    corruption=0.0,  # Disable corruption for conversion
                    enc=tf.nn.tanh, dec=None)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(prev_model["params"], write_version=tf.train.SaverDef.V2)
        if os.path.isfile(prev_model_path):
            saver.restore(sess, prev_model_path)
        X_train = sess.run(prev_model["encode"], feed_dict={prev_model["input"]: X_train})
        X_valid = sess.run(prev_model["encode"], feed_dict={prev_model["input"]: X_valid})
        X_test = sess.run(prev_model["encode"], feed_dict={prev_model["input"]: X_test})
    del prev_model

    reset()

    # Hyperparameters
    learning_rate = 0.0001
    corruption = 0.9
    ae_enc = tf.nn.tanh
    ae_dec = None

    training_iters = 1000
    batch_size = 10
    n_classes = 2

    # Load model
    model = ae(prev_code_size, code_size, corruption=corruption, enc=ae_enc, dec=ae_dec)

    # Use GD for optimization of model cost
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model["cost"])

    # Initialize Tensorflow session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Define model saver
        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)

        # Initialize with an absurd cost for model selection
        prev_costs = np.array([9999999999] * 3)

        # Iterate Epochs
        for epoch in range(training_iters):

            # Break training set into batches
            batches = range(int(len(X_train) / batch_size))
            costs = np.zeros((len(batches), 3))

            for ib in batches:

                # Compute start and end of batch from training set data array
                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                # Select current batch
                batch_xs, batch_ys = X_train[from_i:to_i], y_train[from_i:to_i]

                # Run optimization and retrieve training cost
                _, cost_train = sess.run(
                    [optimizer, model["cost"]],
                    feed_dict={
                        model["input"]: batch_xs
                    }
                )

                # Compute validation cost
                cost_valid = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: X_valid
                    }
                )

                # Compute test cost
                cost_test = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: X_test
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]

            # Compute the average costs from all batches
            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs

            # Pretty print training info
            print (
                "Exp={experiment}, Model=ae2, Iter={epoch:5d}, Cost={cost_train:.6f} {cost_valid:.6f} {cost_test:.6f}",
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "cost_train": cost_train,
                    "cost_valid": cost_valid,
                    "cost_test": cost_test,
                }
            )

            # Save better model if optimization achieves a lower cost
            if cost_valid < prev_costs[1]:
                print ("Saving better model")
                saver.save(sess, model_path)
                prev_costs = costs
            else:
                print


def run_finetuning(experiment,
                   X_train, y_train, X_valid, y_valid, X_test, y_test,
                   model_path, prev_model_1_path, prev_model_2_path,
                   code_size_1=1000, code_size_2=600):
    """

    Run the pre-trained NN for fine-tuning, using first and second autoencoders' weights

    """

    # Hyperparameters
    learning_rate = 0.0005
    dropout_1 = 0.5
    dropout_2 = 0.8
    dropout_3 = 0.3
    initial_momentum = 0.1
    final_momentum = 0.9  # Increase momentum along epochs to avoid fluctiations
    saturate_momentum = 100

    training_iters = 200
    start_saving_at = 20
    batch_size = 10
    n_classes = 2


    if os.path.isfile(model_path) or \
       os.path.isfile(model_path + ".meta"):
        return

    # Convert output to one-hot encoding
    y_train = np.array([to_softmax(n_classes, y) for y in y_train])
    y_valid = np.array([to_softmax(n_classes, y) for y in y_valid])
    y_test = np.array([to_softmax(n_classes, y) for y in y_test])

    # Load pretrained encoder weights
    ae1 = load_ae_encoder(X_train.shape[1], code_size_1, prev_model_1_path)
    ae2 = load_ae_encoder(code_size_1, code_size_2, prev_model_2_path)

    # Initialize NN model with the encoder weights
    model = nn(X_train.shape[1], n_classes, [
        {"size": code_size_1, "actv": tf.nn.tanh},
        {"size": code_size_2, "actv": tf.nn.tanh},
        {"size": 100, "actv": tf.nn.tanh},
    ], [
        {"W": ae1["W_enc"], "b": ae1["b_enc"]},
        {"W": ae2["W_enc"], "b": ae2["b_enc"]},
        {"W": (np.random.randn(600,100)/10000).astype(np.float32), "b": ae2["b_enc"][:100]},
    ])

    # Place GD + momentum optimizer
    model["momentum"] = tf.placeholder("float32")
    optimizer = tf.train.MomentumOptimizer(learning_rate, model["momentum"]).minimize(model["cost"])

    # Compute accuracies
    correct_prediction = tf.equal(
        tf.argmax(model["output"], 1),
        tf.argmax(model["expected"], 1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Initialize Tensorflow session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Define model saver
        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)

        # Initialize with an absurd cost and accuracy for model selection
        prev_costs = np.array([9999999999] * 3)
        prev_accs = np.array([0.0] * 3)

        # Iterate Epochs
        
        for epoch in range(training_iters):

            # Break training set into batches
            batches = range(int(len(X_train) / batch_size))
            costs = np.zeros((len(batches), 3))
            accs = np.zeros((len(batches), 3))

            # Compute momentum saturation
            alpha = float(epoch) / float(saturate_momentum)
            if alpha < 0.:
                alpha = 0.
            if alpha > 1.:
                alpha = 1.
            momentum = initial_momentum * (1 - alpha) + alpha * final_momentum

            total_train=0
            for ib in batches:

                # Compute start and end of batch from training set data array
                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                # Select current batch
                batch_xs, batch_ys = X_train[from_i:to_i], y_train[from_i:to_i]

                # Run optimization and retrieve training cost and accuracy
                _, cost_train, acc_train,y_train_expected,y_train_output = sess.run(
                    [optimizer, model["cost"], accuracy, model["expected"], model["output"]],
                    feed_dict={
                        model["input"]: batch_xs,
                        model["expected"]: batch_ys,
                        model["dropouts"][0]: dropout_1,
                        model["dropouts"][1]: dropout_2,
                        model["dropouts"][2]: dropout_3,
                        model["momentum"]: momentum,
                    }
                )
                
                # Compute validation cost and accuracy
                cost_valid, acc_valid = sess.run(
                    [model["cost"], accuracy],
                    feed_dict={
                        model["input"]: X_valid,
                        model["expected"]: y_valid,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0,
                        model["dropouts"][2]: 1.0,
                    }
                )

                # Compute test cost and accuracy
                cost_test,acc_test,y,y_pred= sess.run(
                    [model["cost"], accuracy, model["expected"], model["output"]],
                    feed_dict={
                        model["input"]: X_test,
                        model["expected"]: y_test,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0,
                        model["dropouts"][2]: 1.0,
                    }
                )

                # if epoch==20:
                #   print(batch_xs.shape)
                #   total_train+=batch_xs.shape[0]
                #   print(y_train_expected,y_train_output)
                  
                costs[ib] = [cost_train, cost_valid, cost_test]
                accs[ib] = [acc_train, acc_valid, acc_test]

            # print(total_train+X_valid.shape[0]+X_test.shape[0])

            # Compute the average costs from all batches
            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs

            # Compute the average accuracy from all batches
            accs = accs.mean(axis=0)
            acc_train, acc_valid, acc_test = accs

            # Pretty print training info
            print (
                "Exp={experiment}, Model=mlp, Iter={epoch:5d}, Acc={acc_train:.6f} {acc_valid:.6f} {acc_test:.6f}, Momentum={momentum:.6f}",
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "acc_train": acc_train,
                    "acc_valid": acc_valid,
                    "acc_test": acc_test,
                    "momentum": momentum,
                }
            )
        
            # Save better model if optimization achieves a lower accuracy
            # and avoid initial epochs because of the fluctuations
            if acc_valid > prev_accs[1] and epoch > start_saving_at:
                print ("Saving better model")
                saver.save(sess, model_path)
                prev_accs = accs
                prev_costs = costs
            else:
                print
          

        #print(y,y_pred)
        return y,y_pred


def run_nn(hdf5, experiment, code_size_1, code_size_2):
    tf.disable_v2_behavior()

    exp_storage = hdf5["experiments"][experiment]

    for fold in exp_storage:
        
        experiment_cv = format_config("{experiment}_{fold}", {
            "experiment": experiment,
            "fold": fold,
        })
       
        #print(hdf5["patients"])
        # print(hdf5["experiments"].keys())
        # for i in hdf5["experiments"]["cc200_whole"]["0"]["train"]:
        #   print(i)
        
        X_train, y_train, \
        X_valid, y_valid, \
        X_test, y_test = load_fold(hdf5["patients"], exp_storage, fold)

        X_all=np.vstack((X_train,X_valid,X_test))
        #np.save('X_NYU.npy',X_all)
        print(X_all.shape)
        

        X_all=X_all[:,:-2]
        y_all=np.concatenate((np.array(y_train),np.array(y_valid),np.array(y_test)),axis=0)
        #np.save('y_NYU.npy',y_all)
        
        print(y_all.shape)

        ks=0
        if X_all.shape[1]<10000:
          ks=1000
        else:
          ks=3000
        X_new=SelectKBest(f_classif,k=ks).fit_transform(X_all, y_all)
        print(X_new.shape)

        train=X_train.shape[0]
        valid=X_valid.shape[0]
        test=X_test.shape[0]

      
        X_train=X_new[:train]
        X_valid=X_new[train:train+valid]
        X_test=X_new[train+valid:train+valid+test]


        X_pheno=np.concatenate((X_new,X_all[:,-2:]),axis=1)
        print(X_pheno.shape)

        X_train_2=X_pheno[:train]
        X_valid_2=X_pheno[train:train+valid]
        X_test_2=X_pheno[train+valid:train+valid+test]

        print(X_test_2.shape)

        #os._exit()

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

        
        X_train,X_test,y_train,y_test = train_test_split(X_new,y,test_size=0.1)
        X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.33)

        print(X_train.shape,X_valid.shape,X_test.shape)'''
        


        
        ae1_model_path = format_config("./data/models/{experiment}_autoencoder-1.ckpt", {
            "experiment": experiment_cv,
        })
        ae2_model_path = format_config("./data/models/{experiment}_autoencoder-2.ckpt", {
            "experiment": experiment_cv,
        })
        nn_model_path = format_config("./data/models/{experiment}_mlp.ckpt", {
            "experiment": experiment_cv,
        })

        reset()

        # Run first autoencoder
        run_autoencoder1(experiment_cv,
                         X_train, y_train, X_valid, y_valid, X_test, y_test,
                         model_path=ae1_model_path,
                         code_size=code_size_1)

        reset()

        # Run second autoencoder
        run_autoencoder2(experiment_cv,
                         X_train, y_train, X_valid, y_valid, X_test, y_test,
                         model_path=ae2_model_path,
                         prev_model_path=ae1_model_path,
                         prev_code_size=code_size_1,
                         code_size=code_size_2)

        reset()

        # Run multilayer NN with pre-trained autoencoders
        y_test,y_pred=run_finetuning(experiment_cv,
                       X_train_2, y_train, X_valid_2, y_valid, X_test_2, y_test,
                       model_path=nn_model_path,
                       prev_model_1_path=ae1_model_path,
                       prev_model_2_path=ae2_model_path,
                       code_size_1=code_size_1,
                       code_size_2=code_size_2)
        return y_test,y_pred

if __name__ == "__main__":

    reset()

    arguments = docopt(__doc__)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    # hdf5 = hdf5_handler("./data/abide.hdf5", "a")
    hdf5 = hdf5_handler(bytes("./data/abide.hdf5",encoding="utf8"), 'a')

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative
                   in arguments["<derivative>"]
                   if derivative in valid_derivatives]

    experiments = []
    #print(derivatives)

    for derivative in derivatives:

        config = {"derivative": derivative}

        if arguments["--whole"]:
            experiments += [format_config("{derivative}_whole", config)]

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

    # First autoencoder bottleneck
    code_size_1 = 1000

    # Second autoencoder bottleneck
    code_size_2 = 600

    experiments = sorted(experiments)

    y_pred=[]
    turn=0
    for experiment in experiments:
 
        print(experiment)
        #y_pred = run_nn(hdf5, experiment, code_size_1, code_size_2)
        if len(y_pred)==0:
          y_test,y_pred=run_nn(hdf5, experiment, code_size_1, code_size_2)
        else:
          _,pred=run_nn(hdf5, experiment, code_size_1, code_size_2)
          if turn==1:
            y_pred+=0.5*pred
          if turn==2:
            y_pred+=0.7*pred
        turn+=1

    

    # print(np.argmax(y_pred, 1))
    # print(np.argmax(y_test, 1))

    correct_pred=sum(np.equal(np.argmax(y_pred, 1),np.argmax(y_test, 1)))

    print(correct_pred/len(np.argmax(y_test, 1)))
    
    #accuracy = tf.cast(correct_prediction, "float")
    #print('accuracy:',accuracy)
    #print(y_pred)
    #print(y_test)

