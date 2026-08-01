<div>

# State Space Models and Kalman Filtering for Time Series Analysis {#state-space-models-and-kalman-filtering-for-time-series-analysis .p-name}

</div>

::: {.section .p-summary field="subtitle"}
Techniques for understanding the hidden states of time series data
:::

::::::: {.section .e-content field="body"}
:::::: {#7b09 .section .section .section--body .section--first .section--last}
::: section-divider

------------------------------------------------------------------------
:::

:::: section-content
::: {.section-inner .sectionLayout--insetColumn}
### State Space Models and Kalman Filtering for Time Series Analysis {#0b6d .graf .graf--h3 .graf--leading .graf--title name="0b6d"}

#### Techniques for understanding the hidden states of time series data {#b495 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="b495"}

State space models analyze time series by modeling the underlying,
unobserved states that generate observable data. The Kalman filter, a
cornerstone of this approach, provides an elegant solution for
estimating these hidden states in real time. This article explores the
theoretical foundations and practical implementations of these methods,
showcasing their versatility in various applications.

<figure id="4abc" class="graf graf--figure graf-after--p">
<img
src="https://cdn-images-1.medium.com/max/800/1*3FA4cH1094Be3Lh5JGTa3Q.gif"
class="graf-image" data-image-id="1*3FA4cH1094Be3Lh5JGTa3Q.gif"
data-width="1500" data-height="1000" data-is-featured="true" />
<figcaption>Kalman filter in action</figcaption>
</figure>

### Mathematical Foundation of State Space Models {#57a6 .graf .graf--h3 .graf-after--figure name="57a6"}

State Space Models describe dynamic systems through a pair of equations.
The State Transition Equation, also known as the process model,
mathematically captures how system states change from one time step to
the next, incorporating both deterministic dynamics and process noise.
This equation forms the core of predicting system behavior and
represents the internal dynamics that may not be directly observable.

The Observation Equation, also called the measurement model, establishes
the mathematical relationship between the hidden system states and the
measurements that can actually be observed or measured. This equation is
crucial because it allows us to infer information about the internal
states from external measurements, accounting for measurement noise and
sensor characteristics.

These equations form the basis for implementing various state estimation
techniques, from simple Kalman Filters to more complex nonlinear
estimators. This mathematical structure provides a powerful and flexible
way to model and analyze dynamic systems across numerous applications.

``` {#e127 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
import numpy as np

class StateSpaceModel:
    def __init__(self, state_dim, observation_dim):
        # State transition matrix (F)
        self.F = np.eye(state_dim)
        # Observation matrix (H)
        self.H = np.zeros((observation_dim, state_dim))
        self.H[0, 0] = 1
        # Process noise covariance (Q)
        self.Q = np.eye(state_dim) * 0.1
        # Observation noise covariance (R)
        self.R = np.eye(observation_dim) * 1.0
        # Initial state mean and covariance
        self.x0 = np.zeros(state_dim)
        self.P0 = np.eye(state_dim)
```

### The Kalman Filter Algorithm {#7548 .graf .graf--h3 .graf-after--pre name="7548"}

The Kalman filter is a recursive estimation algorithm for linear systems
affected by Gaussian noise distributions. This filter achieves optimal
state estimation by combining model predictions with sensor measurements
in a mathematically rigorous way. The algorithm operates through a
two-step process that continuously refines its estimates as new data
becomes available.

The Prediction step, also known as the time update, uses the system
model to forecast the next state and its associated uncertainty
covariance. This step projects the current state estimate forward in
time according to the known system dynamics, accounting for any control
inputs and process noise. The prediction equations propagate both the
state estimate and its uncertainty to provide a prior estimate for the
next time step.

The Update step, or measurement update, incorporates new sensor
measurements to refine the predicted state estimate. When new
observations become available, the filter computes the Kalman gain --- a
weighting factor that balances the relative uncertainty between
predictions and measurements. This gain is used to optimally combine the
prediction with new measurements, resulting in an improved posterior
state estimate and updated uncertainty covariance. The recursive nature
of these two steps makes the Kalman filter computationally efficient and
suitable for real-time applications.

``` {#e5a8 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
class KalmanFilterCustom:
    def __init__(self, state_space_model):
        self.model = state_space_model
        self.state_dim = len(state_space_model.x0)
        self.x = self.model.x0
        self.P = self.model.P0

def predict(self):
        """Prediction step"""
        self.x = self.model.F @ self.x
        self.P = self.model.F @ self.P @ self.model.F.T + self.model.Q
        return self.x, self.P
    def update(self, measurement):
        """Update step"""
        y = measurement - self.model.H @ self.x  # Innovation
        S = self.model.H @ self.P @ self.model.H.T + self.model.R  # Innovation covariance
        K = self.P @ self.model.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y  # Update state
        self.P = (np.eye(self.state_dim) - K @ self.model.H) @ self.P  # Update covariance
        return self.x, self.P
```

### Practical Example: Tracking a Moving Object {#ba37 .graf .graf--h3 .graf-after--pre name="ba37"}

Object tracking is a simple real-world example of state estimation
techniques. In this example, the true object motion follows a
predictable sinusoidal pattern, but our ability to track it is
complicated by noise in our sensor measurements, making it an excellent
demonstration of filter performance.

The system can be modeled using state space equations where the state
vector includes both position and velocity components. The sinusoidal
trajectory provides a natural test case because it involves continuous
changes in both position and velocity, challenging the filter's ability
to maintain accurate tracking. The state transition model must account
for the underlying sinusoidal dynamics, while the measurement model
reflects the noisy position observations from sensors.

Implementation of this tracking system typically shows how the filter
can effectively remove measurement noise and provide smooth estimates of
the object's true position and velocity. This example particularly
highlights the filter's ability to maintain tracking accuracy even when
measurements are noisy or intermittent, making it a valuable
demonstration of state estimation principles in action. The performance
can be visualized by plotting the true trajectory, noisy measurements,
and filtered estimates, clearly showing how the filter smooths out
measurement noise while maintaining accurate tracking.

``` {#f249 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
def generate_trajectory(n_steps, noise_std=0.1):
    """Generate a noisy trajectory"""
    t = np.linspace(0, 4 * np.pi, n_steps)
    true_position = 10 * np.sin(t)
    true_velocity = 10 * np.cos(t)
    true_states = np.vstack((true_position, true_velocity))
    measurements = true_position + np.random.normal(0, noise_std, n_steps)
    return true_states, measurements

def track_object():
    # Generate data
    n_steps = 100
    true_states, measurements = generate_trajectory(n_steps)
    # Initialize model and filter
    model = StateSpaceModel(state_dim=2, observation_dim=1)
    model.F = np.array([[1, 1], [0, 1]])  # Position and velocity
    kf = KalmanFilterCustom(model)
    # Run filter
    estimated_states = []
    for measurement in measurements:
        kf.predict()
        est_state, _ = kf.update(measurement)
        estimated_states.append(est_state)
    return np.array(estimated_states), true_states, measurements
# Execute tracking
estimated_states, true_states, measurements = track_object()
```

### Advanced Filters for Nonlinear Systems {#7af6 .graf .graf--h3 .graf-after--pre name="7af6"}

Extended Kalman Filter (EKF) linearizes nonlinear models around the
current state estimate, making it possible to apply Kalman filter
principles to nonlinear systems. EKF operates by performing a local
linearization using Taylor series expansion and computing Jacobian
matrices (partial derivatives) to create a linear approximation at the
current operating point. This process involves two main steps:
prediction, where the state is projected ahead using the nonlinear
model, and update, where the linearized model is used for the Kalman
update equations.

The EKF is used in navigation systems, robot localization, target
tracking, process control, and financial modeling. Its ability to handle
nonlinear systems while maintaining relatively efficient computation
makes it a practical choice for many real-world applications.

``` {#9754 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="python"}
class ExtendedKalmanFilter:
    def __init__(self, f, h, q_dim, r_dim):
        self.f = f  # State transition function
        self.h = h  # Measurement function
        self.Q = np.eye(q_dim) * 0.1
        self.R = np.eye(r_dim) * 1.0

def predict(self, x, P):
        x_pred = self.f(x)
        F = self.numerical_jacobian(self.f, x)
        P_pred = F @ P @ F.T + self.Q
        return x_pred, P_pred
    def update(self, x_pred, P_pred, measurement):
        H = self.numerical_jacobian(self.h, x_pred)
        y = measurement - self.h(x_pred)  # Innovation
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
        x = x_pred + K @ y
        P = (np.eye(len(x)) - K @ H) @ P_pred
        return x, P
    @staticmethod
    def numerical_jacobian(func, x, eps=1e-7):
        n = len(x)
        J = np.zeros((n, n))
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            J[:, i] = (func(x_plus) - func(x)) / eps
        return J
```

Unscented Kalman Filter (UKF) uses carefully chosen sigma points to
estimate and propagate the mean and covariance of system states through
nonlinear transformations, avoiding the need for explicit Jacobian
calculations required by EKF. The UKF operates by selecting a set of
sample points (sigma points) around the current state estimate,
propagating these points through the nonlinear system, and then
reconstructing the transformed mean and covariance from the propagated
points. This process involves two main steps: generating and propagating
sigma points through the nonlinear model, and reconstructing the
statistical properties from the transformed points.

The UKF is used in similar applications as EKF, including navigation,
target tracking, and state estimation for autonomous vehicles, but tends
to perform better in systems with stronger nonlinearities. Its ability
to capture higher-order statistical moments while maintaining reasonable
computational complexity makes it increasingly preferred over EKF in
many modern applications.

``` {#41fb .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="scss"}
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

def initialize_ukf(dim_x, dim_z, fx, hx):
    points = MerweScaledSigmaPoints(dim_x, alpha=0.1, beta=2.0, kappa=-1)
    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=1.0, fx=fx, hx=hx, points=points)
    ukf.Q = np.eye(dim_x) * 0.1
    ukf.R = np.eye(dim_z) * 1.0
    return ukf
```

### Practical Applications and Visualization {#110f .graf .graf--h3 .graf-after--pre name="110f"}

Here's how to visualize the results of the Kalman filter:

``` {#6681 .graf .graf--pre .graf-after--p .graf--preV2 code-block-mode="1" spellcheck="false" code-block-lang="scss"}
import matplotlib.pyplot as plt

def visualize_results(true_states, measurements, estimated_states):
    plt.figure(figsize=(12, 6))
    plt.plot(true_states[0], label="True Position")
    plt.plot(measurements, 'r.', label="Measurements")
    plt.plot(estimated_states[:, 0], 'g-', label="Estimated Position")
    plt.title("Kalman Filter Tracking")
    plt.legend()
    plt.grid(True)
    plt.show()
# Visualize
visualize_results(true_states, measurements, estimated_states)
```

Key considerations in selecting and implementing state estimation
filters center on matching the filter type to the system characteristics
and noise properties. For linear systems with Gaussian noise, the
standard Kalman Filter provides optimal estimation. However, nonlinear
systems require more advanced approaches like EKF or UKF, while systems
with non-Gaussian noise distributions are best handled by Particle
Filters. This systematic approach to model selection ensures the most
effective estimation strategy for a given application.

Parameter tuning forms a crucial part of filter implementation, with
particular attention needed for the process noise covariance (Q) and
measurement noise covariance (R) matrices. These parameters
significantly influence filter performance and must be carefully
calibrated to reflect the actual system and measurement uncertainties. Q
represents the uncertainty in the system model, while R represents the
uncertainty in sensor measurements.

Performance evaluation typically relies on metrics such as Root Mean
Square Error (RMSE), which quantifies the difference between estimated
and true states. RMSE provides a standardized way to assess filter
accuracy and compare different implementations, helping engineers
optimize their filter designs and ensure reliable state estimation. This
metric is particularly valuable during the testing and validation phases
of filter development.

### Conclusion {#cf15 .graf .graf--h3 .graf-after--p name="cf15"}

State space models and Kalman filtering are powerful tools for time
series analysis, offering robust solutions to noisy measurements and
hidden states. Whether tackling linear or nonlinear systems,
understanding the trade-offs between accuracy, complexity, and
computational requirements is essential for successful implementation.
:::
::::
::::::
:::::::

By [Kyle Jones](https://medium.com/@kylejones_47003){.p-author .h-card}
on [January 17, 2025](https://medium.com/p/df404ad4cc2b).

[Canonical
link](https://medium.com/@kylejones_47003/state-space-models-and-kalman-filtering-for-time-series-analysis-df404ad4cc2b){.p-canonical}

Exported from [Medium](https://medium.com) on February 9, 2025.
