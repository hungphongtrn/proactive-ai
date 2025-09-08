# Unifying Reward and Preference: A Hybrid Verifiable-Preference Optimization

We propose a new hybrid objective, which we will call **Verifiable Advantage and Preference Optimization Ratio (VAPOR)**. This approach maintains a single-loss structure while seamlessly embedding both verifiable reward and preference signals through a unified probability ratio.

Let's first define our variables for clarity:
-   $p$: The input prompt.
-   $s_i$: The verifiable part of the generation (`REASONING`, `INTENT`, `EMOTION`) on which the reward function is applied.
-   $(y_w, y_l)$: The pair of preferred (winner) and dispreferred (loser) responses associated with the prompt $p$ and verifiable output $s_i$.

The core of our proposal is a new, preference-aware probability ratio, $r'_{i(\theta)}$, which combines both verifiable reward-based and preference-based optimization signals.

## 1. The Hybrid Probability Ratio

The refined probability ratio $r'_i(\theta)$ is a product of two terms: a verifiable reward term that captures structured generation quality and a preference term that incorporates human feedback.

It is formulated as follows:

$$r'_i(\theta) = \underbrace{\left( \frac{\pi_\theta(s_i|p)}{\pi_{\text{ref}}(s_i|p)} \right)}_{\text{Verifiable Reward Term}} \cdot \underbrace{\exp\left( \beta \left( \log \frac{\pi_\theta(y_w|p, s_i)}{\pi_{\text{ref}}(y_w|p, s_i)} - \log \frac{\pi_\theta(y_l|p, s_i)}{\pi_{\text{ref}}(y_l|p, s_i)} \right) \right)}_{\text{Preference Term}}$$

Let's deconstruct the novel **Preference Term**:

-   **Log-Probability Ratios**: The terms $\log\left(\frac{\pi_\theta(y|p, s_i)}{\pi_{\text{ref}}(y|p, s_i)}\right)$ measure how much the current policy $\pi_\theta$ has shifted the log-probability of generating a response $y$ compared to the reference policy $\pi_{\text{ref}}$.
-   **Preference Difference**: We take the difference between the log-ratio for the *winning* response ($y_w$) and the log-ratio for the *losing* response ($y_l$).
    -   If this difference is positive, it means the policy is correctly increasing the likelihood of $y_w$ relative to $y_l$, which is the desired behavior.
    -   If it is negative, the policy is incorrectly favoring the dispreferred response.
-   **Exponentiation**: We wrap the entire expression in $\exp(\cdot)$ to convert the log-domain difference back into a non-negative, multiplicative ratio. This makes it compatible with advantage-based optimization, where it will scale the advantage $A_i$.
-   **The $\beta$ Hyperparameter**: This temperature parameter provides explicit control over the strength of the preference signal.
    -   A higher $\beta$ will amplify the effect of the preference data.
    -   Setting $\beta = 0$ makes the entire preference term equal to 1, effectively reducing to pure verifiable reward-based optimization.
    -   This provides fine-grained control over the balance between verifiable reward and preference signals.

## 2. The Final VAPOR Objective Function

With this new hybrid ratio, the final objective function remains a single, unified loss that elegantly balances verifiable reward-based and preference-based learning through structured optimization.

The final **VAPOR** objective is:

$$\mathcal{L}_{\text{VAPOR}}(\theta) = \mathbb{E} \left[ \min \left( r'_i(\theta) A_i, \text{clip} \left( r'_i(\theta), 1-\varepsilon, 1+\varepsilon \right) A_i \right) \right] - w_1 \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{orig}})$$

Here, the advantage $A_i$ is calculated based on the reward from the verifiable output $s_i$, but its application is now scaled by both the verifiable reward-based probability ratio and the preference-based ratio, creating a unified optimization signal that respects both verifiable feedback and human preferences.