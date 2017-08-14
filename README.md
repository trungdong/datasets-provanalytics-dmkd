# Provenance Network Analytics Datasets

This repository provides the datasets used in the Provenance Network Analytics
paper and the code for its analyses. The code was also used to generate the
charts shown in our paper. Please note that the information provided here is
meant to accompany the paper, where the analytic method is described in more
detail.

## Overview

Provenance network analytics is a novel data analytics approach that helps infer
properties of data, such as quality or trustworthiness, from their provenance.
Instead of analysing application data, which are typically domain-dependent, it
analyses the data's provenance as represented using the World Wide Web
Consortium's domain-agnostic [PROV data model](https://www.w3.org/TR/prov-dm/).
Specifically, the approach proposes a number of network metrics (PNM) for
provenance data and applies machine learning techniques over such metrics to
build predictive models for some key properties of data. Applying this method on
the provenance of real-world data from three different applications, we show
that provenance network analytics can successfully identify the owners of
provenance documents, assess the trustworthiness of crowdsourced data, and
identify instructions from chat messages in an alternate-reality game with high
levels of accuracy.

The notebooks and the accompanied datasets provided in this repository
demonstrate how the method can be applied in a number of domains as a useful and
generic tool for data analytics.

## Installation

You do not need to install anything to see the notebooks provided in this repository (linked below). However, if you want to re-run the code on the datasets, you will need to install a number of required Python packages as listed in the [`requirements.txt`](requirements.txt) as shown below.

The code provided with the datasets were run on Python 3.6. However, it might still run on other Python versions, but this is not guaranteed. All the packages required to run the experiments are listed in `requirements.txt`. In order to install those, run the following command with `pip`.
```bash
pip install -r requirements.txt
```

## Provenance Datasets

We use three datasets in our paper, which listed below. Each dataset contains a
number of provenance graphs and their labels. Instead of providing the actual
provenance graphs, due to privacy issues, we only provide here the provenance
network metrics calculated from those graphs (which are used in our analyses).

1. Provenance documents on [ProvStore](https://provenance.ecs.soton.ac.uk/store/):
    - [`provstore/data.csv`](provstore/data.csv): the PNM of provenance documents uploaded to ProvStore and their corresponding owners (anonymised as u_1, u_2, ...)
2. Provenance of [CollabMap](https://collabmap.org/) data:
    - [`collabmap/trust_values.csv`](collabmap/trust_values.csv): the trust value of each data entity from CollabMap (identified by the `id` column).
    - [`collabmap/depgraphs.csv`](collabmap/depgraphs.csv): the PNM of the _provenance dependency graph_ of each data entity. (See our paper for the definition of a provenance dependency graph)
    - [`collabmap/ancestor-graphs.csv`](collabmap/ancestor-graphs.csv): the PNM of the provenance graph of each data entity.
3. Provenance from the [Radiation Response Game](https://dx.doi.org/10.1007/978-3-319-06498-7_4) (RRG).
    - `rrg/depgraphs-k.csv`, e.g. [`rrg/depgraphs-5.csv`](rrg/depgraphs-5.csv): the PNM of the provenance dependency graph level _k_ of a RRG chat message (k = 1..15).


## IPython Notebooks

The notebooks below provide the code for the analysis of the above datasets as reported in our paper. They detail the steps we took in our experiments and also show their results.

- Application 1: [Identifying the owner of a provenance document](Application%201%20-%20ProvStore%20Documents.ipynb)
- Application 2: [Assessing the trustworthiness of crowdsourced data in CollabMap](Application%202%20-%20CollabMap%20Data%20Quality.ipynb)
- Application 3: [Identifying instructions from chat messages in the Radiation Response Game](Application%203%20-%20RRG%20Messages.ipynb)

In addition, we also provide here extra materials to help with replicating the
experiments and to document extra experiments we carried out, which are not
included in the paper due to space constraints.

- [Common cross validation test code](Cross%20Validation%20Code.ipynb): explaining our evaluation method as implemented in [`analytics.py`](analytics.py) (and used in the three above notebooks).
- [Extra 1 - Comparing machine learning algorithms](Extra%201%20-%20Comparing%20ML%20algorithms.ipynb): we compared the performance of a number of classifiers provided by the [scikit-learn package](http://scikit-learn.org/stable/) over our datasets in terms of accuracy and time.
- Extra 2: we compare the performance of the decision tree classifiers on
_unbalanced_ datasets v.s. _balanced_ ones. Note that we did not balance data
in Application 3 as they are already fairly balanced.
    + [Extra 2.1 - Unbalanced Data - Application 1](Extra%202.1%20-%20Unbalanced%20Data%20-%20Application%201.ipynb)
    + [Extra 2.2 - Unbalanced Data - Application 2](Extra%202.1%20-%20Unbalanced%20Data%20-%20Application%202.ipynb)
