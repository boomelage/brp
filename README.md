

**Machine Learning Option Pricing:**  
An empirical approach based on S&P500 data


# Table of Contents
1. [Introduction](#introduction)
2. [Model Specification](#model-specification)
   - [Pricing model](#pricing-model)
3. [Calibration Approach](#calibration-approach)
   - [Approximation of implied volatilities](#approximation-of-implied-volatilities)
   - [Interpolation of volatility surface](#interpolation-of-volatility-surface)
   - [Calibration data generation](#calibration-data-generation)
4. [Training Procedure](#training-procedure)
   - [Data generation](#data-generation)
   - [Machine learning model specification](#machine-learning-model-specification)
5. [Results](#results)
6. [Proposal for Future Research](#proposal-for-future-research)
7. [Conclusions](#conclusions)
   - [Notes on calibration](#notes-on-calibration)
   - [Preliminary testing](#preliminary-testing)
8. [References](#references)


# Introduction

The main scope of this paper is to devise a data generation method that can be reliably used to train machine learning algorithms in pricing options under stochastic volatility models. It has been demonstrated by Frey et al. (2022) that Single Layer and Deep Neural networks are able to accurately predict option prices given a large synthetic training dataset. 

The data generation method outlined below is an extension of Frey et al. (2022) with additional considerations regarding the feasibility of all feature combinations. The machine learning model specification is nearly identical to that of Frey et al. (2022) with minor exceptions including the use of relative absolute pricing error as a performance metric, as well as exploration of additional activation functions and solvers.

Following this result, it was considered whether SkLearn Neural Network models could be used to price exotic options such as barrier and Asian options. These options often require stochastic volatility models due to their path-dependent payoffs. Since exotic options lack closed-form solutions, they are computationally intensive to price, especially for a portfolio of derivatives.

Thus, we aim to generalize a data generation routine in which a set of Heston model parameters is calibrated and used to price a variety of options based on historical spot prices and at-the-money volatilities for historical option data. To achieve this, we implement several approximations, introducing various assumptions and limitations.


# Model Specification

## Pricing model

To model the logarithmic price of the underlying security, we use the Heston (1993) model, defined by the following pair of stochastic differential equations:

```math
dX_t = \left( r - \frac{v_t}{2} \right) dt + \sqrt{v_t} \left( \rho dW_t + \sqrt{1 - \rho^2} dB_t \right),
```

```math
dv_t = \alpha (\beta - v_t) dt + \eta \sqrt{v_t} dW_t,

```

Derman's approximation of implied volatilites:
```math
\sigma(K, t_0) = \sigma_{\text{atm}}(S_0, t_0) - b(t_0)(K - S_0),
```
