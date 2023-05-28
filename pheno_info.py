#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Describe phenotypical info

Usage:
  pheno_info.py
  pheno_info.py (-h | --help)

Options:
  -h --help           Show this screen

"""

from docopt import docopt
from utils import (load_phenotypes,load_phenotypes_2)
import os


if __name__ == "__main__":

    arguments = docopt(__doc__)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    #pheno_path = "./data/phenotypes/Phenotypic_V1_0b.csv"
    pheno = load_phenotypes_2(pheno_path)

    for site, df_site in pheno.groupby("SITE_ID"):
        asd = df_site[df_site["DX_GROUP"] == 0]
        tc = df_site[df_site["DX_GROUP"] == 1]
        fmt = "% 8s &    % 8.1f (% 8.1f) & % 8.1f (% 8.1f) & % 8.1f (% 8.1f)   &  M % 3d, F % 3d &    % 8.1f (% 8.1f) &  M % 3d, F % 3d \\\\"

        print (fmt % (
            site,
            asd["AGE"].mean(),
            asd["AGE"].std(),
            asd["HANDEDNESS_SCORES"].mean(),
            asd["HANDEDNESS_SCORES"].std(),
            asd["FIQ"].mean(),
            asd["FIQ"].std(),
            int(asd[asd["SEX"] == "M"].shape[0]),
            int(asd[asd["SEX"] == "F"].shape[0]),
            tc["AGE"].mean(),
            tc["AGE"].std(),
            int(tc[tc["SEX"] == "M"].shape[0]),
            int(tc[tc["SEX"] == "F"].shape[0]),
        ))


    os._exit()

    for site, df_site in pheno.groupby("SITE_ID"):
        asd = df_site[df_site["DX_GROUP"] == 0]
        tc = df_site[df_site["DX_GROUP"] == 1]
        fmt = "% 8s &    % 8.1f (% 8.1f) & % 8.1f (% 8.1f)    &  M % 3d, F % 3d &    % 8.1f (% 8.1f) &  M % 3d, F % 3d \\\\"

        print (fmt % (
            site,
            asd["AGE"].mean(),
            asd["AGE"].std(),
            #int(0),
            #int(0),
            #asd["ADOS"].mean(),
            #asd["ADOS"].std(),
            asd["MEAN_FD"].mean(),
            asd["MEAN_FD"].std(),
            int(asd[asd["SEX"] == "M"].shape[0]),
            int(asd[asd["SEX"] == "F"].shape[0]),
            tc["AGE"].mean(),
            tc["AGE"].std(),
            int(tc[tc["SEX"] == "M"].shape[0]),
            int(tc[tc["SEX"] == "F"].shape[0]),
        ))
