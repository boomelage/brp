**Machine Learning Option Pricing:**  
An empirical approach based on market data

# Table of Contents
1. [Introduction](#introduction)
2. [Model Specification](#model-specification)
   * 2.1 [Pricing Model](#pricing-model)
   * 2.2 [Volatility Estimation](#volatility-estimation)
   * 2.3 [Calibration Procedure](#calibration-procedure)
3. [References](#references)


# 1. Introduction

The main scope of this paper is to devise a data generation method that can be reliably used to train machine learning algorithms in pricing options under stochastic volatility. It has been demonstrated by Frey et al. (2022) that Single Layer and Deep Neural networks are able to accurately predict vanilla option prices given a large synthetic training dataset. 

The data generation method proposed is an extension of that in Frey et al. (2022) with additional considerations regarding the feasibility of all feature combinations. The machine learning model specification is nearly identical to that of Frey et al. (2022) with minor exceptions including the use of relative absolute pricing error against relative moneyness as a performance metric, as well as exploration of additional parameterisation.

Following this eniqury into the felixbility of this method, it was considered whether ScikitLearn Neural Network models could be used to price path-dependent exotic options. Thus, we aim to generalize a data generation routine utilising generic at-the-money volatilites which will allow us to train the model on data strictly adhering to desired market conditions while allowing for accurately calibrated stochastic volatility for each trading day simulated in part.


# 2. Model Specification
## 2.1 Pricing Model

To model the logarithmic price of our underlying security, we use the Heston (1993) model, described by the pair of stochastic differential equations:

```math
dX_t = \left( r - \frac{v_t}{2} \right) dt + \sqrt{v_t} \left( \rho dW_t + \sqrt{1 - \rho^2} dB_t \right),
```

```math
dv_t = \kappa (\theta - v_t) dt + \eta \sqrt{v_t} dW_t
```
where
- $v_0$ represents the initial variance,
- $\theta$ is the long-run variance,
- $\rho$ is the correlation between the log-price process and its volatility,
- $\kappa$ is the mean reversion of the variance to **𝜃**,
- $\eta$ is the volatility of the variance process, and 
- $B_t$ , $W_t$ are continuous random walks. 

## 2.2 Volatility Estimation
The model becomes suitable for fitting to our proposed method via approximation of implied volatilities as proposed by Derman (2008):
```math
\sigma(K, t_0) = \sigma_{\text{atm}}(S_0, t_0) - b(t_0)(K - S_0)
```
## 2.3 Calibration Procedure
To calibrate our Heston (1993) model for a given trading day, we begin by estimating $b$ coefficients by extracting the term structure of volatility for each maturity where an at-the-money volatility is present and applying the above regression.
<br>

# 3. References
Blanda, V. (2023). FX Barrier Option Pricing. Available at: https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/212252650---VALENTIN-BLANDA---BLANDA_VALENTIN_02293988.pdf

Derman, E. (2008). Lecture 9: Patterns of Volatility Change. Available at: https://emanuelderman.com/wp-content/uploads/2013/09/smile-lecture9.pdf 

Frey, C., Scheuch, C., Voigt, S. and Weiss, P. (2022). Option Pricing via Machine Learning with Python. Tidy Finance. 
Available at: https://www.tidy-finance.org/python/option-pricing-via-machine-learning.html

Heston, S. (1993). A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.
Available at: https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf
