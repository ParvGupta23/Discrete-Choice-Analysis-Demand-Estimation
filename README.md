# Discrete Choice Analysis & Demand Estimation

## Project Overview
This project implements econometric models to analyze consumer behavior and risk frequency. It focuses on **Multinomial Logit (MNL)** models for discrete choice (brand preference) and **Poisson Regression** for count data (event incidence).

The codebase features a custom **Maximum Likelihood Estimation (MLE)** pipeline built in MATLAB, designed to quantify price elasticities, marginal effects, and regional risk factors.

## Key Features
* **Multinomial Logit Solver:** Implements a Newton-Raphson compatible optimization for MNL models using the Log-Sum-Exp trick for numerical stability.
* **Marginal Effects Engine:** Calculates own-price and cross-price elasticities to evaluate market share sensitivity to pricing shocks.
* **Risk Modeling:** Utilizes Poisson GLMs with log-link functions to forecast event frequencies based on regional and demographic covariates.
* **Scenario Forecasting:** Generates out-of-sample probability forecasts for market share (MNL) and safety thresholds (Poisson).

## Methodology
### 1. Brand Choice (Conditional Logit)
We model the utility $U_{ij}$ of individual $i$ choosing alternative $j$ as:
$$P_{ij} = \frac{\exp(x_{ij}'\beta)}{\sum_{k=1}^{J} \exp(x_{ik}'\beta)}$$
* **Data:** Panel data of consumer choices across 3 major beverage brands.
* **Optimization:** Minimized Negative Log-Likelihood via Quasi-Newton algorithms.

### 2. Count Data Analysis (Poisson)
We model the expected count of events $\lambda_i$ as:
$$\ln(\lambda_i) = \beta_0 + \beta_1 \ln(\text{Exposure}) + \beta_2 \text{Covariate}$$
* **Application:** Analyzed regional incident frequency to determine risk factors associated with inventory density.

## Usage
1.  **Brand Choice:** Run `mnl_brand_choice.m` to estimate price sensitivity and forecast market shares.
2.  **Risk Analysis:** Run `poisson_risk_model.m` to generate GLM statistics and risk probability tables.

## Technologies
* **Language:** MATLAB
* **Libraries:** Optimization Toolbox, Statistics and Machine Learning Toolbox
