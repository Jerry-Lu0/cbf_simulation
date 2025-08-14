import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cvxpy as cp
import math

class WaypointCBFRobot:
    def __init__(self, length=2, x=0, y=0, theta=np.pi/2, v=0.2, w=0.1):
        self.length = length
        self.x = x
        self.y = y
        self.v = v
        self.w = w
        self.theta = theta
        
        # Waypoint following parameters
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_tolerance = 2.0  # Distance to consider waypoint reached
        self.max_speed = 2.0
        self.lookahead_distance = 3.0
        
        # PID parameters for waypoint following
        self.kp_linear = 1.0
        self.kp_angular = 2.0
        self.kd_angular = 0.5
        self.prev_angular_error = 0.0
        
        # Track control history for analysis
        self.control_history = {
            'desired': [],
            'actual': [],
            'cbf_active': [],
            'solver_status': [],
            'waypoint_progress': [],
            'constraint_violations': []
        }
        
    def set_waypoints(self, waypoints):
        """Set list of waypoints [(x1, y1), (x2, y2), ...]"""
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        
    def get_current_waypoint(self):
        """Get current target waypoint"""
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None
    
    def distance_to_waypoint(self, waypoint):
        """Calculate distance to a waypoint"""
        return math.sqrt((self.x - waypoint[0])**2 + (self.y - waypoint[1])**2)
    
    def angle_to_waypoint(self, waypoint):
        """Calculate desired heading angle to reach waypoint"""
        return math.atan2(waypoint[1] - self.y, waypoint[0] - self.x)
    
    def update_waypoint_progress(self):
        """Check if current waypoint is reached and update target"""
        current_wp = self.get_current_waypoint()
        if current_wp is None:
            return False
            
        distance = self.distance_to_waypoint(current_wp)
        if distance < self.waypoint_tolerance:
            print(f"Reached waypoint {self.current_waypoint_idx + 1}: {current_wp}")
            self.current_waypoint_idx += 1
            return True
        return False
    
    def compute_waypoint_control(self):
        """Compute desired control inputs for waypoint following"""
        current_wp = self.get_current_waypoint()
        if current_wp is None:
            return np.array([0.0, 0.0])  # Stop if no more waypoints
            
        # Distance and angle to waypoint
        distance = self.distance_to_waypoint(current_wp)
        desired_angle = self.angle_to_waypoint(current_wp)
        
        # Angle error (normalized to [-pi, pi])
        angle_error = desired_angle - self.theta
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
            
        # Desired linear velocity (proportional to distance, capped at max_speed)
        desired_v = min(self.kp_linear * distance, self.max_speed)
        
        # Slow down when turning
        if abs(angle_error) > math.pi/4:
            desired_v *= 0.5
            
        # Desired angular velocity (PID control)
        angular_error_derivative = angle_error - self.prev_angular_error
        desired_w = self.kp_angular * angle_error + self.kd_angular * angular_error_derivative
        desired_w = max(min(desired_w, 1.5), -1.5)  # Clamp angular velocity
        
        self.prev_angular_error = angle_error
        
        return np.array([desired_v, desired_w])

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
        # Get desired control from waypoint following
        u_des = self.compute_waypoint_control()
        
        # Define optimization variables
        u = cp.Variable(2)
        
        # Objective function
        objective = cp.Minimize(0.5 * cp.sum_squares(u - u_des))
        
        # Constraints
        constraints = []
        
        # Control input bounds
        constraints.append(u[0] >= 0.0)
        constraints.append(u[0] <= 2.0)
        constraints.append(u[1] >= -1.0)
        constraints.append(u[1] <= 1.0)
        
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
        
        control_result = {
            'desired': u_des.copy(),
            'actual': u_des.copy(),
            'solver_status': 'unknown',
            'cbf_active': False,
            'constraint_info': cbf_constraints,
            'emergency_mode': False,
            'waypoint_reached': False,
            'current_waypoint': self.get_current_waypoint()
        }
        
        # Update waypoint progress
        control_result['waypoint_reached'] = self.update_waypoint_progress()
        
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
        self.control_history['waypoint_progress'].append(self.current_waypoint_idx)
        
        return control_result

    def get_direction_arrow(self):
        arrow_length = self.length * 0.8
        start = np.array([self.x, self.y])
        end = start + arrow_length * np.array([np.cos(self.theta), np.sin(self.theta)])
        return start, end

    def get_waypoint_direction_arrow(self):
        """Show direction to current waypoint"""
        current_wp = self.get_current_waypoint()
        if current_wp is None:
            return np.array([self.x, self.y]), np.array([self.x, self.y])
            
        arrow_length = self.length * 0.6
        start = np.array([self.x, self.y])
        wp_angle = self.angle_to_waypoint(current_wp)
        end = start + arrow_length * np.array([np.cos(wp_angle), np.sin(wp_angle)])
        return start, end


# Create animation with enhanced visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Initialize robot with waypoints
robot = WaypointCBFRobot(x=-15, y=-15, theta=np.pi/4, v=2.0)

# Define waypoints (similar to your original waypoint structure but adapted to this coordinate system)
waypoints = [
    (-15, -15),  # 起点
    (-12, -12),
    (-6,   2),
    (  0,   8),
    ( 10,  10),
    ( 15,   0),
    (  0, -20),
    (-10,  -8),
    ( -2,   0)   # 终点
]

robot.set_waypoints(waypoints)

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
# Waypoint direction (what robot wants to do)
waypoint_arrow, = ax1.plot([], [], 'lime', linewidth=3, alpha=0.7, linestyle='--', label='waypoint direction')

center_point, = ax1.plot([], [], 'ko', markersize=10, label='robot center')

# Plot waypoints
waypoint_x = [wp[0] for wp in waypoints]
waypoint_y = [wp[1] for wp in waypoints]
ax1.plot(waypoint_x, waypoint_y, 'mo', markersize=8, alpha=0.7, label='waypoints')
ax1.plot(waypoint_x, waypoint_y, 'm--', alpha=0.5, linewidth=1)

# Add waypoint numbers
for i, (x, y) in enumerate(waypoints):
    ax1.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='magenta', alpha=0.7),
                fontsize=8, fontweight='bold', color='white')

# Current waypoint highlight
current_waypoint_marker, = ax1.plot([], [], 'ro', markersize=15, alpha=0.8, label='current target')

trajectory_x = []
trajectory_y = []
trajectory_line, = ax1.plot([], [], 'g-', alpha=0.5, linewidth=2, label='trajectory')

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
ax1.set_title("CBF Waypoint Following Controller", fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')

# Control analysis panel (right panel)
ax2.set_xlim(0, 1000)
ax2.set_ylim(-2, 7)
ax2.grid(True, alpha=0.3)
ax2.set_title("Control Analysis: Waypoint Following + CBF", fontsize=14, fontweight='bold')
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
    'waypoints_reached': 0,
    'infeasible': 0
}

def init():
    for line in lines.values():
        line.set_data([], [])
    actual_arrow.set_data([], [])
    waypoint_arrow.set_data([], [])
    center_point.set_data([], [])
    current_waypoint_marker.set_data([], [])
    trajectory_line.set_data([], [])
    desired_v_line.set_data([], [])
    actual_v_line.set_data([], [])
    desired_w_line.set_data([], [])
    actual_w_line.set_data([], [])
    info_text.set_text('')
    return (list(lines.values()) + walls + safety_walls + 
            [actual_arrow, waypoint_arrow, center_point, current_waypoint_marker, 
             trajectory_line, info_text, desired_v_line, actual_v_line, desired_w_line, actual_w_line])

def update(frame):
    global counters
    
    # Apply CBF control with waypoint following
    control_result = robot.apply_cbf_qp(x_min, x_max, y_min, y_max, d_safe)
    
    # Update counters
    counters['total'] += 1
    if control_result['cbf_active']:
        counters['cbf_active'] += 1
    if control_result['emergency_mode']:
        counters['emergency'] += 1
    if control_result['waypoint_reached']:
        counters['waypoints_reached'] += 1
    if control_result['solver_status'] == cp.INFEASIBLE:
        counters['infeasible'] += 1
    
    # Record trajectory
    trajectory_x.append(robot.x)
    trajectory_y.append(robot.y)
    
    if len(trajectory_x) > 1000:
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
    
    # Waypoint direction arrow
    wp_start, wp_end = robot.get_waypoint_direction_arrow()
    waypoint_arrow.set_data([wp_start[0], wp_end[0]], [wp_start[1], wp_end[1]])
    
    center_point.set_data([robot.x], [robot.y])
    
    current_wp = robot.get_current_waypoint()
    if current_wp is not None:
        current_waypoint_marker.set_data([current_wp[0]], [current_wp[1]])
    else:
        current_waypoint_marker.set_data([], [])
    
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
    
    # Calculate distance to current waypoint
    current_wp = robot.get_current_waypoint()
    wp_distance = robot.distance_to_waypoint(current_wp) if current_wp else 0
    
    # Status information
    status_color = "🟢" if not control_result['emergency_mode'] else "🔴"
    cbf_status = "🟡 ACTIVE" if control_result['cbf_active'] else "⚪ INACTIVE"
    wp_status = f"WP {robot.current_waypoint_idx + 1}/{len(robot.waypoints)}" if current_wp else "COMPLETE"
    
    info_text.set_text(
        f'{status_color} Frame: {frame}\n'
        f'Pos: ({robot.x:.2f}, {robot.y:.2f})\n'
        f'θ: {robot.theta:.2f}\n'
        f'Waypoint: {wp_status}\n'
        f'WP Dist: {wp_distance:.2f}\n'
        f'Desired: v={control_result["desired"][0]:.2f}, ω={control_result["desired"][1]:.2f}\n'
        f'Actual: v={robot.v:.2f}, ω={robot.w:.2f}\n'
        f'CBF: {cbf_status}\n'
        f'Solver: {control_result["solver_status"]}\n'
        f'Min h-value: {min_cbf:.2f} ({critical_constraint[0]})\n'
        f'Emergency: {counters["emergency"]}/{counters["total"]}\n'
        f'CBF Active: {counters["cbf_active"]}/{counters["total"]}\n'
        f'Waypoints: {counters["waypoints_reached"]}/{len(robot.waypoints)}'
    )
    
    return (list(lines.values()) + walls + safety_walls + 
            [actual_arrow, waypoint_arrow, center_point, current_waypoint_marker,
             trajectory_line, info_text, desired_v_line, actual_v_line, desired_w_line, actual_w_line])

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=5000, init_func=init, blit=True, interval=50
)

plt.tight_layout()
plt.show()