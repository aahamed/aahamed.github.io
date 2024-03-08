+++
title = 'Maximum Likelihood Estimation'
date = 2023-11-19T17:48:00-08:00
math = true
+++

If you have consumed machine learning content you may be familiar with the idea of training models to learn patterns by feeding it lots of data. A key component in the training process is the error or loss function, which gives the model an objective to minimize. This post will go over how to derive the loss function using a technique called Maximum Likelihood Estimation ( MLE ). It is a fascinating technique, that sits at the core of machine learning science, and a great example of how people have used mathematics to solve some really interesting and challenging problems.

### Problem Statement
Suppose you are given N points drawn i.i.d from a Gaussian distribution with unknown parameters $\mu$ and $\sigma$. Could you estimate the parameters from the observed data?

### Solution

We start by re-phrasing the question as: Which parameters  are most likely given the observed data? Mathematically this can be expressed as:
$$
\argmax_{\theta} P(\theta | D)
$$
Here, $\theta = [\mu, \sigma]$ are the parameters and $D$ is the N points drawn from the gaussian distribution.

Trying to maximize $P(\theta | D)$ directly is hard, so we can rewrite this in terms of $P(D | \theta)$ using Bayes rule:
$$
\begin{aligned}
\argmax_{\theta} P(\theta | D) &= \argmax_{\theta} \dfrac{P(D | \theta) P(\theta)}{P(D)} \quad \quad \text{(Bayes Rule)} \\\\
\argmax_{\theta} \mathcal{L} &= \argmax_{\theta}  P(D | \theta)
\end{aligned}
$$

Here $P(D | \theta)$ is the likelihood of the data given the parameters ( aka $\mathcal{L}$). And $P(\theta)$ and $P(D)$ are constants which don't affect the argmax operation.

Substituting $\{x_1, ..., x_n\}$ for $D$ and $\{\mu, \sigma\}$ for $\theta$ we get:
$$
\begin{aligned}
\mathcal{L} &= P(x_1, ..., x_n | \mu, \sigma) \\\\
&= P(x_1 | \mu, \sigma) P (x_2, ..., x_n |  \mu, \sigma) \quad \quad \text{(Product rule and Conditional Independence)}\\\\
&= \prod_{i=1}^{N} P(x_i | \mu, \sigma)
\end{aligned}
$$

Since we want to choose the parameters that maximize the likelihood of the observed data, we should find the argmax of $\mathcal{L}$ w.r.t to the parameters ${\mu, \sigma}$:
$$
\argmax_{\mu, \sigma} \mathcal{L} = \argmax_{\mu, \sigma} \prod_{i=1}^{N} P(x_i | \mu, \sigma)
$$
To make the maximization easier we can take the log of the likelihood ( it is easier to take the derivative of a sum than a product ):
$$
\begin{aligned}
\argmax_{\mu, \sigma} \log \mathcal{L} &= \argmax_{\mu, \sigma} \log \prod_{i=1}^{N} P(x_i | \mu, \sigma) \\\\
&= \argmax_{\mu, \sigma} \sum_{i=1}^{N} \log P(x_i | \mu, \sigma) \quad \quad \text{log product rule}
\end{aligned}
$$

Many texts will also multiply the log likelihood by -1 and change the argmax to argmin:
$$
\argmax_{\mu, \sigma} \mathcal{L} = \argmin_{\mu, \sigma} - \log \mathcal{L}
$$
The $- \log \mathcal{L}$ is also called the error or loss function.

Because samples are drawn i.i.d ( independently and identically distributed ) the likelihood $\mathcal{L}$ can be decomposed as the product of the likelihood of the individual samples. Since each sample is drawn from the gaussian distribution, the probability of a single sample $x_i$ is:
$$
P(x_i | \mu, \sigma) = \dfrac{1}{\sqrt{2\pi\sigma^2}} \exp\left[ -\dfrac{(x_i - \mu)^2}{2\sigma^2}\right]
$$

After substituting this in our expression for the negative log likelihood and finding the argmin ( writing this out in latex is extremely tedious ), MLE will give the following estimates for $\mu$ and $\sigma$ :
$$
\begin{aligned}
\mu &= \dfrac{1}{N} \sum_{i}^{N} x_i \\\\
\sigma &= \sqrt{\frac{1}{n}\sum_{i=1}^{n} (x_i - \mu)^2}
\end{aligned}
$$

Here we see the estimate for $\mu$ is the sample mean and the estimate for $\sigma$ is the sample standard deviation.

**Restating the original problem:**

Suppose you are given N points drawn i.i.d from a Gaussian distribution with unknown parameters $\mu$ and $\sigma$. Could you estimate the parameters from the observed data?

**Solution:**

If we use MLE, we would estimate $\mu$ by calculating the sample mean and estimate $\sigma$ by calculating the sample standard deviation.

## Modeling Functions with MLE

### Problem Statement

Suppose you have a conditional distribution $P(y|x)$ where $y = h(x) + \epsilon$, where h is some polynomial function and $\epsilon$ is 0-mean gaussian noise.  Now, suppose you are given N points drawn i.i.d from $P(y|x)$. How can you estimate the function $h(x)$?

### Solution

We don't have direct access to $h(x)$ so we model it using a parametrized function $f(x, \theta)$ where $\theta$ are the parameters. We can then choose the parameters that maximize the likelihood of the observed data:

$$
\begin{aligned}
\mathcal{L} &= \prod_{i} P(y_i | x_i) \\\\
-\log \mathcal{L} &=  -\sum_i \log P(y_i | x_i)
\end{aligned}
$$

Since $y = h(x) + \epsilon$, we can write $P(y | x)$ as:

$$
P(y | x) = \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left[ - \frac{(y - f(x, \theta))^2}{2 \sigma^2} \right]
$$

Plugging this in to our expression for negative log likelihood and taking the argmin, we get:
$$
\argmin - \log \mathcal{L} = \sum_i (y_i - f(x_i, \theta))^2
$$

The above expression is also known as the sum squared error. 

### Logistic Regression

The above solution is great when we want to model real-valued functions but won't work for binary classification.

### Problem Statement

Suppose you have 2 classes $y \in \\{ C_0=0, C_1=1\\}$ and a conditional distribution 
$$
P(y|x) =
\begin{cases}
  p(x) & \text{if } y=1 \\\\
  1-p(x) & \text{if } y=0
\end{cases}
$$
where $p(x)$ is a probability in $[0, 1]$.

We don't have direct access to $p(x)$, so we model it using the parametrized logistic function
$$
h(x; \theta) = \frac{1}{1 + e^{-\theta^{\intercal}x + b}} \in [0, 1]
$$ 

Now, suppose you are given N points drawn i.i.d from $P(y|x)$. How can you estimate $p$ ?

### Solution

As before, we want to choose the parameter $\theta$ that maximizes the likelihood of the observed data:

$$
\begin{aligned}
\mathcal{L} &= \prod_{i} P(y_i | x_i) \\\\
-\log \mathcal{L} &=  -\sum_i \log P(y_i | x_i)
\end{aligned}
$$

We can rewrite $P(y|x)$ as:
$$
P(y_i=t_i | x_i) = h(x_i; \theta)^{t_i}(1-h(x_i; \theta))^{1-t_i}
$$

Here $t_i$ is the outcome $\\{1, 0\\}$ of the $i^{th}$ point in the observed data.

Substituting this into the expression for MLE gives:
$$
\begin{aligned}
-\log \mathcal{L} &=  -\sum_i \log h(x_i; \theta)^{t_i}(1-h(x_i; \theta))^{1-t_i} \\\\
&= - \sum_i t^i \log h(x_i; \theta) + (1-t_i)\log(1-h(x_i; \theta))
\end{aligned}
$$

The above expression is also known as the binary cross entropy error. By taking the argmin of the above expression we can find the paramter $\theta$ that maximizes the likelihood of the observed data. Plugging this value of $\theta$ back into the logistic function $h(x; \theta)$ allows us to get an estimate of $p(x)$.

### Multinomial Regression

The above solution is great for binary classification but won't work for multinomial classification ( more than 2 classes ).

### Problem Statement

Suppose you have K classes $y \in \\{ C_0, C_1, ..., C_{K-1}\\}$ where each class $C_i$ is represented as a one-hot encoded vector. The targets follow the conditional distribution:
$$
P(y=C_k | x) = p_k(x)
$$
where $p_k(x)$ is a probability in $[0, 1]$. Hence $p(x)$ is a vector of probabilities of dimension $k$.

We don't have direct access to $p_k(x)$, so we model it using the parametrized softmax function:
$$
h^k(x; \theta) = \frac{\exp(\theta_k^{\intercal}x)}{\sum_{j=1}^{K} \exp(\theta_j^{\intercal}x)}
$$

The softmax distribution is just a generalization of the logistic function from binary classification. If $K=2$, then softmax function decomposes into the logistic function.

Now, suppose you are given N points drawn i.i.d from $P(y|x)$. How can you estimate $p(x)$?

### Solution

As before, we want to choose the parameter $\theta$ that maximizes the likelihood of the observed data:

$$
\begin{aligned}
\mathcal{L} &= \prod_{i} P(y_i | x_i) \\\\
-\log \mathcal{L} &=  -\sum_i \log P(y_i | x_i)
\end{aligned}
$$

We can rewrite $P(y_i | x_i)$ as:

$$
P(y_i=t_i | x_i) = \prod_k (h_i^k)^{t_i^k}
$$

Here $t_i$ is the one hot encoding of the observed class $C_k$ of the $i^{th}$ datapoint.

Substituting in this back to MLE expression:
$$
\begin{aligned}
-\log \mathcal{L} &=  -\sum_i \log \prod_k (h_i^k)^{t_i^k} \\\\
&= - \sum_i \sum_k t_i^k \log h_i^k
\end{aligned}
$$

The above expression is known as cross entropy error. By taking the argmin of the above expression w.r.t $\theta$, we can find the paramters that maximizes the likelihood of the observed data. Plugging this value of $\theta$ back into the softmax function $h(x; \theta)$ allows us to get an estimate of $p(x)$.


## Recap

Maximum Likelihood Estimation is a technique for finding model parameters that best fit the observed data. As the model designer, you still need to choose the right distribution for the data you are trying to model:
- If the targets in your data is from gaussian distribution, then MLE will result in sum-squared error
- If the targets are from the bernoulli distribution, then MLE will result in binary cross entropy error
- If the targets are from the multinomial distribution, then MLE will result in cross entropy error

If you made it this far, thanks for reading :)