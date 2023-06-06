# Autism-Spectrum-Disorder-Diagnosis-with-Ensemble-DNN

**Paper Title: Autism Spectrum Disorder Diagnosis with Ensemble Deep
Neural Networ**

**Abstract:** The objective of this study is to formulate an effective approach for early Autism Spectrum Disorder (ASD) detection utilizing brain functional magnetic resonance imaging (fMRI) data. We present a weighted ensemble learning architecture using an innovative DNN, feature selection and transfer learning technique. 



## Environment Setup

Please be aware that this code is meant to be run with Python 3 under Linux (MacOS may work, Windows probably not). Download the packages from requirements.txt:

```bash
!pip install -r requirements.txt
```
We use the [ABIDE I dataset](http://fcon_1000.projects.nitrc.org/indi/abide/). We use the pre-processing method from the [preprocessed-connectomes-project](https://github.com/preprocessed-connectomes-project/abide).


## Run Experiments

We put all the experiment command and the result in the Experiments.ipynb file. As an example, we prepare the hdf5 files in advance of the cc200, aal,ez data from the NYU site, and show the entire experiment on this data as followings:

1. Run download_abide.py to download the raw data.

We use the command below:

```
#download_abide.py [--pipeline=cpac] [--strategy=filt_global] [<derivative> ...]

!python /content/drive/MyDrive/acerta-abide/download_abide.py
```

2. To show the demographic Information of ABIDE I.

```bash
!python pheno_info.py
```

3. Run prepare_data.py to compute the correlation. Then we can get the hdf5 files. The dataset(hdf5) is put on the [colab link](https://drive.google.com/file/d/1-WyQ7IOqSxaGcoA6MR4ydJdazlqzKYMY/view?usp=drive_link), you need to put it in the "data" folder. Or you can simply run the code as:

```
#download_abide.py [--folds=N] [--whole] [--male] [--threshold] [--leave-site-out] [--NYU-site-out] [<derivative> ...]

!python prepare_data.py --leave-site-out cc200 aal ez
```

5. Using Stacked Sparse Denoising Autoencoder (SSDAE) to perform Multi-atlas Deep Feature Representation, and using multilayer perceptron (MLP) and ensemble learning to classify the ASD and TC.

```
#nn.py [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]

!python nn.py --leave-site-out cc200 aal ez
```  

4. Evaluating the MLP model on test dataset. You can use the saved models from provious or download the models from the [link](https://drive.google.com/drive/folders/1rIZpXdafzI-nb0YonkL0XQs6pOSSxRUf?usp=drive_link), and run this command directly without running previous steps.

```
#nn_evaluate.py [--whole] [--male] [--threshold] [--leave-site-out] [--NYU-site-out] [<derivative> ...]

!python nn_evaluate.py --leave-site-out cc200 aal ez
```
        
## Qualify Analysis and Visualization

All the analysis for the subset and the visualization of top ROIs, which mentioned in our work, are included in the **Qualify Analysis and Visualization.ipynb** file.






