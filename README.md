

**Machine Learning Option Pricing:**  
An empirical approach based on market data


# Table of Contents
1. [Introduction](#introduction)
2. [Model Specification](#model-specification)
3. [References](#references)


# 1. Introduction

The main scope of this paper is to devise a data generation method that can be reliably used to train machine learning algorithms in pricing options under stochastic volatility models. It has been demonstrated by Frey et al. (2022) that Single Layer and Deep Neural networks are able to accurately predict option prices given a large synthetic training dataset. 

The data generation method outlined below is an extension of Frey et al. (2022) with additional considerations regarding the feasibility of all feature combinations. The machine learning model specification is nearly identical to that of Frey et al. (2022) with minor exceptions including the use of relative absolute pricing error as a performance metric, as well as exploration of additional activation functions and solvers.

Following this result, it was considered whether SkLearn Neural Network models could be used to price exotic options such as barrier and Asian options. These options often require stochastic volatility models due to their path-dependent payoffs. Since exotic options lack closed-form solutions, they are computationally intensive to price, especially for a portfolio of derivatives.

Thus, we aim to generalize a data generation routine in which a set of Heston model parameters is calibrated and used to price a variety of options based on historical spot prices and at-the-money volatilities for historical option data. To achieve this, we implement several approximations, introducing various assumptions and limitations.


# 2. Model Specification

## Pricing model

To model the logarithmic price of the underlying security, we use the Heston (1993) model, defined by the following pair of stochastic differential equations:

```math
dX_t = \left( r - \frac{v_t}{2} \right) dt + \sqrt{v_t} \left( \rho dW_t + \sqrt{1 - \rho^2} dB_t \right)
```

```math
dv_t = \alpha (\beta - v_t) dt + \eta \sqrt{v_t} dW_t

```

Derman's approximation of implied volatilites:
```math
\sigma(K, t_0) = \sigma_{\text{atm}}(S_0, t_0) - b(t_0)(K - S_0)
```

## 3. References
Derman, E. (2008). Available at: https://emanuelderman.com/wp-content/uploads/2013/09/smile-lecture9.pdf 

Frey, C., Scheuch, C., Voigt, S. and Weiss, P. (2022). Option Pricing via Machine Learning with Python. Tidy Finance. Available at: https://www.tidy-finance.org/python/option-pricing-via-machine-learning.html
