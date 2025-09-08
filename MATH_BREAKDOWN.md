# A Novel Approach to Post-Training LLMs: Preference-Guided Group Relative Policy Optimization (PG-GRPO)

This document outlines the mathematical foundations of two prominent methods for post-training Large Language Models (LLMs): Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO). It then introduces a novel method, Preference-Guided Group Relative Policy Optimization (PG-GRPO), which combines the strengths of both approaches.

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

## **3. Proposed Method: Preference-Guided Group Relative Policy Optimization (PG-GRPO)**

PG-GRPO aims to integrate the direct preference signal from DPO into the robust, on-policy optimization framework of GRPO. The core idea is to replace the probability ratio within GRPO's clipped surrogate loss with a term derived from the DPO loss, thereby directly incorporating preference data into the advantage calculation.

## **3.1. The PG-GRPO Objective Function**

The proposed PG-GRPO objective function maintains the overall structure of the GRPO objective but modifies the clipped surrogate loss:

$$\mathcal{L}_{\text{PG-GRPO}}(\theta) = \mathcal{L}_{\text{pref\_clip}}(\theta) - w_1 \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{orig}})$$

## **3.2. The Preference-Guided Clipped Surrogate Loss ($\mathcal{L}_{\text{pref\_clip}}$)**

The novel component is the `Preference-Guided Clipped Surrogate Loss`. Instead of a simple probability ratio of a single response, we introduce a term that reflects the model's adherence to the learned preferences. We define a "preference ratio" inspired by the DPO formulation.

First, let's define the implicit reward from DPO for a single response $y$ given a prompt $x$ and a reference policy $\pi_{\text{ref}}$:

$$r_{\text{DPO}}(y, x) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

The DPO loss aims to maximize the difference $r_{\text{DPO}}(y_w, x) - r_{\text{DPO}}(y_l, x)$.

In our PG-GRPO, for each sample $i$ in a batch (which can be a single response from a group in GRPO's setting), we can compute a "preference score" that is analogous to the DPO reward. We then use this to form a new ratio for the clipped surrogate loss.

Let's define a **Preference Ratio**, $p_i(\theta)$, which replaces the original probability ratio $r_i(\theta)$:

$$ p_i(\theta) = \exp \left( \beta \left( \log \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)} \right) \right) $$

This formulation is influenced by the DPO's focus on the log-probability ratio. Here, $\pi_{\theta_{old}}$ serves a similar role to $\pi_{\text{ref}}$ in DPO.

The new **Preference-Guided Clipped Surrogate Loss** is then:

$$\mathcal{L}_{\text{pref\_clip}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left( \min \left( p_i(\theta) A_i, \text{clip} ( p_i(\theta), 1-\varepsilon, 1+\varepsilon ) A_i \right) \right)$$

where $A_i$ is the advantage calculated using the GRPO methodology (i.e., the reward of response $i$ minus the average reward of the group).

## **3.3. How PG-GRPO Works**

1.  **Group-Based Sampling and Advantage Estimation:** For a given prompt, the current policy $\pi_\theta$ generates a group of responses. An external reward model (or another scoring function) evaluates each response, and the advantage $A_i$ for each response is calculated relative to the group's average reward, as in standard GRPO.

2.  **Preference-Informed Policy Update:** During the optimization step, the policy update is driven by the $\mathcal{L}_{\text{pref\_clip}}$ loss. This loss considers not just the advantage of a response but also the "preference ratio" $p_i(\theta)$.

    *   If a response has a high advantage ($A_i > 0$), the objective will be to increase $p_i(\theta)$, which in turn increases the probability of generating that response under the new policy.
    *   If a response has a low advantage ($A_i < 0$), the objective will be to decrease $p_i(\theta)$.

3.  **KL Regularization:** The KL penalty term from GRPO is retained to ensure that the policy does not deviate too drastically from the original policy, maintaining training stability.

By integrating a DPO-style ratio into the GRPO framework, PG-GRPO aims to achieve a more direct and potentially more sample-efficient way of learning from preferences, while still benefiting from the stable, on-policy optimization and memory efficiency of GRPO. This hybrid approach allows the model to learn not just from absolute rewards but also from the relative preference between different types of outputs.