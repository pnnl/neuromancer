- DPC as a Safety Layer for PPO
    In this approach, PPO is the primary control policy that interacts with the environment to maximize long-term rewards. DPC acts as a safety layer that monitors PPO’s control actions and corrects them if they violate constraints or if the system is predicted to become unstable.
    How it works:
    PPO generates control actions based on its learned policy, focusing on maximizing long-term rewards through exploration and learning.
    DPC monitors PPO’s actions in real time. Using a neural network, DPC predicts the future states of the system over a short horizon. If PPO’s proposed action leads to unsafe or suboptimal behavior (e.g., violating constraints or causing instability), DPC overrides PPO’s action with a safer one.
    Fallback mechanism: If PPO’s action is safe and within the acceptable range, it is used. If not, DPC’s optimal control action is applied instead.
- DPC for Real-Time Control, PPO for Long-Term Policy Learning
    In this approach, DPC is used for immediate predictive control, ensuring that the system adheres to constraints and is optimized by immediate feedback. PPO is responsible for learning the long-term control policy, helping the system adapt to changes and improve its performance over time. The control policy learned by PPO can guide or enhance the decisions made by DPC.
    How it works:
    DPC handles short-term optimization: At each time step, DPC uses a neural network to predict the system's future states over a short horizon and computes the optimal control action that minimizes a cost function while respecting constraints.
    PPO updates the long-term policy: Over time, PPO learns a control policy that maximizes cumulative rewards by interacting with the environment. PPO can provide feedback to DPC in the form of improved control actions or policy adjustments.
    Policy blending: You can blend the control policies from PPO and DPC by weighting them.
- PPO for Model Learning in DPC
    In this approach, PPO is used to improve the neural network model used in DPC. While DPC typically relies on a pre-trained neural network to predict future states, PPO can continuously update and refine this model based on its interactions with the environment.
    How it works:
    DPC predicts short-term states using a neural network, and it computes optimal control actions based on these predictions.
    PPO refines the neural network model: As PPO interacts with the environment, it improves its understanding of the system’s dynamics. PPO can then update the neural network used by DPC, making the predictions more accurate and improving DPC’s control performance.
    Online learning: PPO continuously learns from real-time data, allowing DPC to adapt to changing system dynamics, external disturbances, or shifts in the environment.
    - [ ] train DPC first, and then train the critic network using DPC policy, and use the DPC policy as initialization of PPO policy
    - [ ] Use NSSM to predict the next model state and add it as extra input to the policy model.