# VAPORTrainer Implementation Plan

This document outlines a phased action plan for implementing the `VAPORTrainer` in `vapor_trainer.py`. The plan synthesizes the algorithmic principles from `VAPOR.md` with the architectural patterns from `trl/trl/trainer/grpo_trainer.py`.

## Phase 1: Core Class Structure and Initialization

**Objective:** Establish the `VAPORTrainer` class, its inheritance, and its configuration.

1.  **Create `vapor_trainer.py`:** Create the file `trl/trl/trainer/vapor_trainer.py`.
2.  **Define `VAPORTrainer` class:**
    *   The class `VAPORTrainer` should inherit from `GRPOTrainer`.
    *   Define the `__init__` method.
3.  **Extend `__init__`:**
    *   The `__init__` method should accept all the arguments of `GRPOTrainer`.
    *   Add the following VAPOR-specific arguments, as detailed in `VAPOR.md`:
        *   `num_generations (int)`
        *   `verifiable_term_split (Tuple[str])`
        *   `preference_term_split (Tuple[str])`
        *   `beta (float)`
    *   Store these arguments as instance attributes (e.g., `self.num_generations`).

## Phase 2: Loss Calculation and Core Logic

**Objective:** Implement the main `compute_loss` method, which is the heart of the VAPOR algorithm.

1.  **Implement `_find_and_map_token_indices`:**
    *   Add the helper function `_find_and_map_token_indices` to the `VAPORTrainer` class as a private method.
    *   The implementation should follow the logic provided in `VAPOR.md`.
2.  **Override `compute_loss`:**
    *   Define the `compute_loss` method within `VAPORTrainer`, overriding the parent method.
    *   This method will orchestrate the entire VAPOR loss calculation.
3.  **Implement Generation Step:**
    *   Inside `compute_loss`, for each prompt in the batch, generate `self.num_generations` completions using the policy model.
4.  **Calculate Log Probabilities:**
    *   Perform forward passes to get log probabilities for all generated completions from both the policy and reference models.
    *   Perform separate forward passes for the static `chosen` and `rejected` sequences from the input batch to get their log probabilities.
5.  **Process Verifiable Term:**
    *   Loop through each generated completion.
    *   Use `_find_and_map_token_indices` to find the verifiable part.
    *   If successful, calculate the reward and the `verifiable_ratio`.
    *   If it fails, set `reward = 0.0` and `verifiable_ratio = 1.0`.
6.  **Calculate Group-Wise Advantage:**
    *   Reshape the collected rewards into a tensor of shape `(batch_size, num_generations)`.
    *   Calculate the mean reward for each group.
    *   Compute the advantage for each completion: `advantages = rewards_tensor - mean_rewards`.
7.  **Calculate Preference Term:**
    *   For each prompt, use the helper function on the `chosen` and `rejected` text.
    *   If successful, calculate the `preference_term` using the DPO-style log-ratios.
    *   If it fails, set `preference_term = 1.0`.
8.  **Assemble Final Loss:**
    *   Combine the `verifiable_ratios` and `preference_terms` to get `hybrid_ratios`.
    *   Calculate the PPO clipped surrogate objective using `hybrid_ratios` and `advantages`.
    *   Add the KL penalty.

## Phase 3: Integration with TRL Infrastructure

**Objective:** Ensure seamless integration with TRL's existing components.

1.  **Data Handling:**
    *   Ensure the data loading and processing can handle the `prompt`, `chosen`, and `rejected` fields.
    *   Leverage the existing `DataLoader` and `Dataset` infrastructure from `GRPOTrainer`.
2.  **Model and Optimizer:**
    *   The trainer should work with models, optimizers, and schedulers in the same way as `GRPOTrainer`. No major changes are expected here.
3.  **Accelerator and Distributed Training:**
    *   Pay close attention to tensor shapes and device placement when implementing the loss calculation to ensure compatibility with `accelerate`.
    *   Use `accelerate.gather` for collecting rewards and other metrics across devices before calculating group-wise statistics.
4.  **Logging and Checkpointing:**
    *   Integrate VAPOR-specific metrics (e.g., mean reward, mean preference term, verifiable ratio) into the logging mechanism.
    *   Ensure that the trainer's state, including VAPOR-specific configurations, is correctly saved and loaded during checkpointing.

## Phase 4: Testing Strategy

**Objective:** Develop a robust testing suite to validate the implementation.

1.  **Unit Tests:**
    *   Write a unit test for `_find_and_map_token_indices` with various edge cases (e.g., missing tags, tags in the wrong order).
    *   Write unit tests for the reward calculation and advantage normalization logic.
    *   Write a unit test for the preference term calculation.
2.  **Integration Tests:**
    *   Create a small-scale integration test that runs the `VAPORTrainer` for a few steps on a mock dataset and a small model.
    *   Verify that the loss is computed without errors and that the model weights are updated.
    *   Check that the logs contain the expected VAPOR-specific metrics.

## Phase 5: Addressing Potential Challenges

**Objective:** Proactively identify and plan for potential implementation hurdles.

1.  **Distributed Training:**
    *   **Challenge:** Ensuring correct group-wise advantage calculation when a group's generations are split across multiple devices.
    *   **Solution:** Use `accelerate.gather` to collect all rewards for a batch onto a single process before calculating the mean and advantages. Then, scatter the results back to the respective devices.
2.  **Memory Management:**
    *   **Challenge:** Generating multiple completions per prompt can be memory-intensive.
    *   **Solution:** Implement micro-batching within the `compute_loss` function if necessary. Process the generated completions in smaller chunks to reduce peak memory usage.
3.  **Numerical Stability:**
    *   **Challenge:** `exp()` operations in ratio calculations can lead to instability.
    *   **Solution:** Perform calculations in log space as much as possible. Add epsilon values to denominators to prevent division by zero.

This phased plan provides a clear roadmap for implementing the `VAPORTrainer`. I will now create a todo list to track the progress of this plan.