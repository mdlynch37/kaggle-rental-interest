# Predicting Rental Listing Interest
__Capstone Project for Udacity’s Machine Learning Nanodegree__

## Software
Python 3.5.3 with a Conda environment exported to `environment.yaml`.
To setup the environment, install Conda from [here][3] and follow [these instructions][4] to create the environment.

## Dataset
The dataset is available on the [Kaggle competition page][0]. A login is required to accept their terms & conditions.

Download, unzip and move`train.json` and `test.json` files into the Data directory.

## Overview
[RentalHop][1] is an online apartment rental listing for the New York City area. One of its differentiating features is its relevancy score, a “HopScore”, by which it sorts listings by default. They would also like to use data on rental properties to improve their product in other ways, like fraud detection and quality control. For this, Two Sigma, their data-focused managing investors, have partnered with Kaggle to hold a machine learning competition: [Two Sigma Connect: Rental Listing Inquiries][2].

RentalHop has back-end functions that could be improved with reliable predictions of how much interest individual listings will generate. These functions are:
- Fraud identification
- Quality control
- Guiding owners and agents toward better listings

By applying a variety of machine learning techniques on rental listing data (price, location, etc.), an algorithm can “learn” complex patterns that correspond to levels of interest users will have in different listings. This algorithm can them provide reliable predictions of how much interest new listings will generate.

[0]:https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data
[1]:https://www.renthop.com/
[2]:https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries

[3]:https://conda.io/docs/installation.html
[4]:https://conda.io/docs/using/envs.html#use-environment-from-file
