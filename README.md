# 🧠 CBF-Based Safe Motion Control Simulation

This project implements a **Control Barrier Function (CBF)** safety filter combined with **Quadratic Programming (QP)** for real-time collision avoidance, and an **Emergency Steering** fallback mechanism. All components are implemented in Python and optionally integrated with ROS 2.

---

## 🚧 1. Control Barrier Function (CBF)

The CBF acts as an **invisible safety fence**, ensuring the robot avoids collisions. It works by defining a safety margin `d_safe` and evaluating a constraint function `h(x)` for each wall:

- If `h > 0` → safe  
- If `h ≤ 0` → risk of collision

We compute the Lie derivative:γ h(x) + L_f h(x) + L_g h(x) · u ≥ 0
This condition adjusts robot control commands `u = [v, w]` in real time to ensure safety.

---

## 📐 2. Quadratic Programming (QP)

### 🎯 Goal:
Find a control `u = [v, w]` that is **closest to the desired command** `u_des` while satisfying all CBF constraints.

### ⚙️ Implementation:
- Solver: [CVXPY](https://www.cvxpy.org/) + [OSQP](https://osqp.org/)
- Objective:  min 0.5 * ‖u - u_des‖²
### ✅ Advantages:
- No manual mode switching
- Far from walls → original `u_des` passes through
- Near walls → QP adjusts `u` smoothly to ensure safety

---

## 🆘 3. Emergency Turn Mechanism

When the QP becomes infeasible (e.g. corner trap), an emergency routine is triggered:

1. **Fix forward speed** to 0.5 m/s
2. **Measure remaining space** in each direction
3. **Rotate** toward the direction with most open space at `w = ±1.0`
4. Resume QP optimization once safe
---

## 🧪 Dependencies

- `numpy`
- `cvxpy`
- `osqp`
- `matplotlib`
- (optional) `rclpy` for ROS 2 integration

---
