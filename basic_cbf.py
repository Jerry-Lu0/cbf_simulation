import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cvxpy as cp

class Robot:
    def __init__(self, length=2, x=0, y=0, theta=np.pi/2, v=0.2, w=0.1):
        self.length = length
        self.x = x
        self.y = y
        self.v = v
        self.w = w
        self.theta = theta
        
        # Track control history for analysis
        self.control_history = {
            'desired': [],
            'actual': [],
            'cbf_active': [],
            'solver_status': [],
            'constraint_violations': []
        }
        
    def get_rotation_matrix(self, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    def get_edges(self):
        l = self.length
        local_edges = {
            'front': np.array([[-l, l], [l, l]]),    
            'back':  np.array([[-l, -l], [l, -l]]),  
            'left':  np.array([[-l, l], [-l, -l]]),  
            'right': np.array([[l, l], [l, -l]])     
        }
        R = self.get_rotation_matrix(self.theta)
        global_edges = {}
        for key, edge in local_edges.items():
            rotated_edge = np.array([R @ point for point in edge])
            global_edges[key] = rotated_edge + np.array([self.x, self.y])
        return global_edges

    def move(self, dt=0.1):
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.w * dt

    def emergency_turn(self, x_min, x_max, y_min, y_max, d_safe):
        """Emergency steering when QP truly fails"""
        current_x, current_y = self.x, self.y
        dx, dy = np.cos(self.theta), np.sin(self.theta)
        
        # Calculate available space
        space_right = (x_max - d_safe) - current_x
        space_left = current_x - (x_min + d_safe)
        space_up = (y_max - d_safe) - current_y
        space_down = current_y - (y_min + d_safe)
        
        # Turn toward largest space
        max_space = max(space_right, space_left, space_up, space_down)
        
        if max_space == space_right and dx <= 0:
            return 1.0
        elif max_space == space_left and dx >= 0:
            return -1.0
        elif max_space == space_up and dy <= 0:
            return 1.0 if dx > 0 else -1.0
        elif max_space == space_down and dy >= 0:
            return -1.0 if dx > 0 else 1.0
        
        return 0.0

    def get_cbf_values(self, x_min, x_max, y_min, y_max, d_safe, beta):
        """Calculate current CBF values for analysis"""
        h_values = []
        
        # x_min boundary
        h1 = (self.x + beta * np.cos(self.theta)) - (x_min + d_safe)
        h_values.append(('x_min', h1))
        
        # x_max boundary  
        h2 = (x_max - d_safe) - (self.x + beta * np.cos(self.theta))
        h_values.append(('x_max', h2))
        
        # y_min boundary
        h3 = (self.y + beta * np.sin(self.theta)) - (y_min + d_safe)
        h_values.append(('y_min', h3))
        
        # y_max boundary
        h4 = (y_max - d_safe) - (self.y + beta * np.sin(self.theta))
        h_values.append(('y_max', h4))
        
        return h_values

    def apply_cbf_qp(self, x_min, x_max, y_min, y_max, d_safe=3.0, gamma=2.0, beta=2.0):
        """Enhanced CBF controller with detailed analysis"""
        # Desired control input (nominal behavior)
        v_des = 2.0
        w_des = 0.0
        u_des = np.array([v_des, w_des])
        
        # Define optimization variables
        u = cp.Variable(2)
        
        # Objective function
        objective = cp.Minimize(0.5 * cp.sum_squares(u - u_des))
        
        # Constraints
        constraints = []
        
        # Control input bounds
        constraints.append(u[0] >= 0.0)
        constraints.append(u[0] <= 6.0)
        constraints.append(u[1] >= -1.5)
        constraints.append(u[1] <= 1.5)
        
        # CBF constraints
        cbf_constraints = []
        
        # x_min boundary
        h1 = (self.x + beta * np.cos(self.theta)) - (x_min + d_safe)
        Lf1 = self.v * np.cos(self.theta) - beta * self.w * np.sin(self.theta)
        Lg1 = np.array([np.cos(self.theta), -beta * np.sin(self.theta)])
        cbf_constraint_1 = gamma * h1 + Lf1 + Lg1 @ u >= 0
        constraints.append(cbf_constraint_1)
        cbf_constraints.append(('x_min', h1, gamma * h1 + Lf1))
        
        # x_max boundary
        h2 = (x_max - d_safe) - (self.x + beta * np.cos(self.theta))
        Lf2 = -self.v * np.cos(self.theta) + beta * self.w * np.sin(self.theta)
        Lg2 = np.array([-np.cos(self.theta), beta * np.sin(self.theta)])
        cbf_constraint_2 = gamma * h2 + Lf2 + Lg2 @ u >= 0
        constraints.append(cbf_constraint_2)
        cbf_constraints.append(('x_max', h2, gamma * h2 + Lf2))
        
        # y_min boundary
        h3 = (self.y + beta * np.sin(self.theta)) - (y_min + d_safe)
        Lf3 = self.v * np.sin(self.theta) + beta * self.w * np.cos(self.theta)
        Lg3 = np.array([np.sin(self.theta), beta * np.cos(self.theta)])
        cbf_constraint_3 = gamma * h3 + Lf3 + Lg3 @ u >= 0
        constraints.append(cbf_constraint_3)
        cbf_constraints.append(('y_min', h3, gamma * h3 + Lf3))
        
        # y_max boundary
        h4 = (y_max - d_safe) - (self.y + beta * np.sin(self.theta))
        Lf4 = -self.v * np.sin(self.theta) - beta * self.w * np.cos(self.theta)
        Lg4 = np.array([-np.sin(self.theta), -beta * np.cos(self.theta)])
        cbf_constraint_4 = gamma * h4 + Lf4 + Lg4 @ u >= 0
        constraints.append(cbf_constraint_4)
        cbf_constraints.append(('y_max', h4, gamma * h4 + Lf4))
        
        # Build and solve QP
        problem = cp.Problem(objective, constraints)
        
        # Analyze what would happen without CBF
        nominal_feasible = True
        if (v_des < 0.0 or v_des > 5.0 or w_des < -1.5 or w_des > 1.5):
            nominal_feasible = False
        
        control_result = {
            'desired': u_des.copy(),
            'actual': u_des.copy(),
            'solver_status': 'unknown',
            'cbf_active': False,
            'constraint_info': cbf_constraints,
            'emergency_mode': False,
            'nominal_feasible': nominal_feasible
        }
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            control_result['solver_status'] = problem.status
            
            if problem.status == cp.OPTIMAL:
                u_opt = u.value
                control_result['actual'] = u_opt.copy()
                
                # Check if CBF significantly modified the control
                control_diff = np.linalg.norm(u_opt - u_des)
                control_result['cbf_active'] = control_diff > 0.1
                
                # Apply the control
                self.v = max(0.0, u_opt[0])
                self.w = u_opt[1]
                
                # Check if solution is essentially "stop and turn"
                if self.v < 0.5:
                    control_result['essentially_stopped'] = True
                else:
                    control_result['essentially_stopped'] = False
                    
            elif problem.status == cp.INFEASIBLE:
                # True infeasibility - use emergency steering
                print(f"QP is infeasible at pos ({self.x:.2f}, {self.y:.2f})")
                emergency_w = self.emergency_turn(x_min, x_max, y_min, y_max, d_safe)
                self.v = 0.5
                self.w = emergency_w
                control_result['emergency_mode'] = True
                control_result['actual'] = np.array([self.v, self.w])
                
            else:
                # Other solver issues
                print(f"QP solver issue: {problem.status}")
                emergency_w = self.emergency_turn(x_min, x_max, y_min, y_max, d_safe)
                self.v = 0.5
                self.w = emergency_w
                control_result['emergency_mode'] = True
                control_result['actual'] = np.array([self.v, self.w])
                
        except Exception as e:
            print(f"QP Exception: {e}")
            emergency_w = self.emergency_turn(x_min, x_max, y_min, y_max, d_safe)
            self.v = 0.5
            self.w = emergency_w
            control_result['emergency_mode'] = True
            control_result['actual'] = np.array([self.v, self.w])
        
        # Store control history
        self.control_history['desired'].append(control_result['desired'])
        self.control_history['actual'].append(control_result['actual'])
        self.control_history['cbf_active'].append(control_result['cbf_active'])
        self.control_history['solver_status'].append(control_result['solver_status'])
        
        return control_result

    def get_direction_arrow(self):
        arrow_length = self.length * 0.8
        start = np.array([self.x, self.y])
        end = start + arrow_length * np.array([np.cos(self.theta), np.sin(self.theta)])
        return start, end

    def get_nominal_direction_arrow(self, v_des=3.0):
        """Show where robot wants to go (nominal behavior)"""
        arrow_length = self.length * 0.6
        start = np.array([self.x, self.y])
        end = start + arrow_length * np.array([np.cos(self.theta), np.sin(self.theta)])
        return start, end


# Create animation with enhanced visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

robot = Robot(x=0, y=0, theta=np.pi/4, v=2.0)

x_min, x_max = -20, 20
y_min, y_max = -20, 20
d_safe = 3

# Main visualization (left panel)
lines = {
    'front': ax1.plot([], [], 'r', linewidth=3, label='robot front')[0],
    'back':  ax1.plot([], [], 'b', linewidth=2)[0],
    'left':  ax1.plot([], [], 'b', linewidth=2)[0],
    'right': ax1.plot([], [], 'b', linewidth=2)[0],
}

# Actual direction (what robot is doing)
actual_arrow, = ax1.plot([], [], 'orange', linewidth=4, alpha=0.8, label='actual direction')
# Nominal direction (what robot wants to do)
nominal_arrow, = ax1.plot([], [], 'lime', linewidth=3, alpha=0.7, linestyle='--', label='desired direction')

center_point, = ax1.plot([], [], 'ko', markersize=10, label='robot center')

trajectory_x = []
trajectory_y = []
trajectory_line, = ax1.plot([], [], 'g-', alpha=0.3, linewidth=1, label='trajectory')

# Boundaries
walls = [
    ax1.plot([x_min, x_max], [y_min, y_min], 'k-', linewidth=3, alpha=0.8)[0],
    ax1.plot([x_min, x_max], [y_max, y_max], 'k-', linewidth=3, alpha=0.8)[0],
    ax1.plot([x_min, x_min], [y_min, y_max], 'k-', linewidth=3, alpha=0.8)[0],
    ax1.plot([x_max, x_max], [y_min, y_max], 'k-', linewidth=3, alpha=0.8)[0],
]

safety_walls = [
    ax1.plot([x_min+d_safe, x_max-d_safe], [y_min+d_safe, y_min+d_safe], 'r:', linewidth=2, alpha=0.6)[0],
    ax1.plot([x_min+d_safe, x_max-d_safe], [y_max-d_safe, y_max-d_safe], 'r:', linewidth=2, alpha=0.6)[0],
    ax1.plot([x_min+d_safe, x_min+d_safe], [y_min+d_safe, y_max-d_safe], 'r:', linewidth=2, alpha=0.6)[0],
    ax1.plot([x_max-d_safe, x_max-d_safe], [y_min+d_safe, y_max-d_safe], 'r:', linewidth=2, alpha=0.6)[0],
]

ax1.set_xlim(-25, 25)
ax1.set_ylim(-25, 25)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title("Enhanced CBF Controller - Robot Behavior", fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')

# Control analysis panel (right panel)
ax2.set_xlim(0, 1000)
ax2.set_ylim(-2, 7)
ax2.grid(True, alpha=0.3)
ax2.set_title("Control Analysis: Desired vs Actual", fontsize=14, fontweight='bold')
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Control Input")

# Control history plots
desired_v_line, = ax2.plot([], [], 'g--', linewidth=2, label='Desired v', alpha=0.7)
actual_v_line, = ax2.plot([], [], 'g-', linewidth=2, label='Actual v')
desired_w_line, = ax2.plot([], [], 'b--', linewidth=2, label='Desired ω', alpha=0.7)
actual_w_line, = ax2.plot([], [], 'b-', linewidth=2, label='Actual ω')

ax2.legend(loc='upper right')

# Info text
info_text = ax1.text(-24, 22, '', fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Counters
counters = {
    'total': 0,
    'cbf_active': 0,
    'emergency': 0,
    'essentially_stopped': 0,
    'infeasible': 0
}

def init():
    for line in lines.values():
        line.set_data([], [])
    actual_arrow.set_data([], [])
    nominal_arrow.set_data([], [])
    center_point.set_data([], [])
    trajectory_line.set_data([], [])
    desired_v_line.set_data([], [])
    actual_v_line.set_data([], [])
    desired_w_line.set_data([], [])
    actual_w_line.set_data([], [])
    info_text.set_text('')
    return (list(lines.values()) + walls + safety_walls + 
            [actual_arrow, nominal_arrow, center_point, trajectory_line, info_text,
             desired_v_line, actual_v_line, desired_w_line, actual_w_line])

def update(frame):
    global counters
    
    # Apply CBF control
    control_result = robot.apply_cbf_qp(x_min, x_max, y_min, y_max, d_safe)
    
    # Update counters
    counters['total'] += 1
    if control_result['cbf_active']:
        counters['cbf_active'] += 1
    if control_result['emergency_mode']:
        counters['emergency'] += 1
    if control_result.get('essentially_stopped', False):
        counters['essentially_stopped'] += 1
    if control_result['solver_status'] == cp.INFEASIBLE:
        counters['infeasible'] += 1
    
    # Record trajectory
    trajectory_x.append(robot.x)
    trajectory_y.append(robot.y)
    
    if len(trajectory_x) > 500:
        trajectory_x.pop(0)
        trajectory_y.pop(0)
    
    # Move robot
    robot.move()
    
    # Update robot visualization
    edges = robot.get_edges()
    for key in lines:
        lines[key].set_data(edges[key][:, 0], edges[key][:, 1])
    
    # Actual direction arrow
    start, end = robot.get_direction_arrow()
    actual_arrow.set_data([start[0], end[0]], [start[1], end[1]])
    
    # Nominal direction arrow (what robot wants to do)
    nominal_start, nominal_end = robot.get_nominal_direction_arrow()
    nominal_arrow.set_data([nominal_start[0], nominal_end[0]], [nominal_start[1], nominal_end[1]])
    
    center_point.set_data([robot.x], [robot.y])
    trajectory_line.set_data(trajectory_x, trajectory_y)
    
    # Update control history plots
    if len(robot.control_history['desired']) > 0:
        time_steps = list(range(len(robot.control_history['desired'])))
        desired_v = [u[0] for u in robot.control_history['desired']]
        actual_v = [u[0] for u in robot.control_history['actual']]
        desired_w = [u[1] for u in robot.control_history['desired']]
        actual_w = [u[1] for u in robot.control_history['actual']]
        
        # Keep only recent history for plotting
        if len(time_steps) > 200:
            time_steps = time_steps[-200:]
            desired_v = desired_v[-200:]
            actual_v = actual_v[-200:]
            desired_w = desired_w[-200:]
            actual_w = actual_w[-200:]
        
        desired_v_line.set_data(time_steps, desired_v)
        actual_v_line.set_data(time_steps, actual_v)
        desired_w_line.set_data(time_steps, desired_w)
        actual_w_line.set_data(time_steps, actual_w)
        
        # Update x-axis for scrolling effect
        if len(time_steps) > 0:
            ax2.set_xlim(max(0, time_steps[-1] - 200), time_steps[-1] + 10)
    
    # Get CBF values for display
    cbf_values = robot.get_cbf_values(x_min, x_max, y_min, y_max, d_safe, beta=2.0)
    min_cbf = min([h for _, h in cbf_values])
    critical_constraint = min(cbf_values, key=lambda x: x[1])
    
    # Status information
    status_color = "🟢" if not control_result['emergency_mode'] else "🔴"
    cbf_status = "🟡 ACTIVE" if control_result['cbf_active'] else "⚪ INACTIVE"
    
    info_text.set_text(
        f'{status_color} Frame: {frame}\n'
        f'Pos: ({robot.x:.2f}, {robot.y:.2f})\n'
        f'θ: {robot.theta:.2f}\n'
        f'Desired: v={control_result["desired"][0]:.2f}, ω={control_result["desired"][1]:.2f}\n'
        f'Actual: v={robot.v:.2f}, ω={robot.w:.2f}\n'
        f'CBF: {cbf_status}\n'
        f'Solver: {control_result["solver_status"]}\n'
        f'Min h-value: {min_cbf:.2f} ({critical_constraint[0]})\n'
        f'Emergency: {counters["emergency"]}/{counters["total"]}\n'
        f'CBF Active: {counters["cbf_active"]}/{counters["total"]}\n'
        f'Stopped: {counters["essentially_stopped"]}/{counters["total"]}'
    )
    
    return (list(lines.values()) + walls + safety_walls + 
            [actual_arrow, nominal_arrow, center_point, trajectory_line, info_text,
             desired_v_line, actual_v_line, desired_w_line, actual_w_line])

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=3000, init_func=init, blit=True, interval=50
)

plt.tight_layout()
plt.show()