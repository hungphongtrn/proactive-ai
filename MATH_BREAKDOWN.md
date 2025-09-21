# Mathematical Breakdown of GRPO and DPO

This document provides a detailed mathematical breakdown of the Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO) algorithms used for fine-tuning large language models (LLMs). Both methods aim to align LLMs with human preferences, but they do so through different approaches and mathematical formulations.

## **1. Background: Group Relative Policy Optimization (GRPO)**

GRPO is a reinforcement learning algorithm designed to fine-tune LLMs by optimizing a policy to generate responses that maximize a given reward function. It is an evolution of Proximal Policy Optimization (PPO) that is more memory-efficient as it does not require a separate value function. Instead, GRPO generates multiple responses for a given prompt and uses the mean reward of this group as a baseline to estimate the advantage of each response.

## **1.1. The GRPO Objective Function**

The core of GRPO lies in its objective function, which aims to maximize the expected advantage of the generated responses while penalizing significant deviations from the original policy. The final objective is composed of two main components: a clipped surrogate loss and a KL divergence penalty.

The final GRPO objective is given by:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathcal{L}_{\text{clip}}(\theta) - w_1 \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{orig}})$$

where:
*   $\mathcal{L}_{\text{clip}}(\theta)$ is the clipped surrogate loss.
*   $w_1$ is the weight of the KL penalty.
*   $\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{orig}})$ is the Kullback-Leibler divergence between the current policy $\pi_\theta$ and the original policy $\pi_{\text{orig}}$.

## **1.2. Clipped Surrogate Loss in GRPO**

The clipped surrogate loss is designed to prevent large, destabilizing policy updates. It is defined as:

$$\mathcal{L}_{\text{clip}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left( \min \left( r_i(\theta) A_i, \text{clip} ( r_i(\theta), 1-\varepsilon, 1+\varepsilon ) A_i \right) \right)$$

where:
*   $N$ is the number of samples.
*   $r_i(\theta) = \frac{\pi_\theta(r_i|p)}{\pi_{\theta_{\text{old}}}(r_i|p)}$ is the probability ratio between the current policy and the old policy for a given response $r_i$ and prompt $p$.
*   $A_i$ is the advantage estimate for the $i$-th sample. In GRPO, this is calculated relative to the group of responses.
*   $\varepsilon$ is a hyperparameter that defines the clipping range.

The advantage $A_i$ is a key component. For a group of responses to a prompt, the advantage of a single response is calculated by comparing its reward to the average reward of the group.

## **2. Background: Direct Preference Optimization (DPO)**

DPO offers a more direct way to align LLMs with human preferences, bypassing the need for an explicit reward model and the complexities of reinforcement learning. DPO directly optimizes the language model on a dataset of preferences, where each data point consists of a prompt and two responses, one preferred ($y_w$) and one dispreferred ($y_l$).

## **2.1. The DPO Loss Function**

The DPO loss function is derived from a theoretical relationship between the optimal policy and a reward function. It is formulated as a binary cross-entropy loss that aims to increase the likelihood of the preferred responses and decrease the likelihood of the dispreferred ones.

The DPO loss is given by:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

where:
*   $\pi_\theta$ is the policy being optimized.
*   $\pi_{\text{ref}}$ is a reference policy (usually the supervised fine-tuned model).
*   $x$ is the prompt, $y_w$ is the preferred response, and $y_l$ is the dispreferred response.
*   $\mathcal{D}$ is the dataset of preferences.
*   $\beta$ is a hyperparameter that controls the temperature.
*   $\sigma$ is the sigmoid function.

Essentially, DPO reframes the preference learning problem as a classification task, making the optimization process more stable and straightforward.