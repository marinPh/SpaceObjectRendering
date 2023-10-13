"""
Author:     Andrew Lawrence Price, Tim Vaughan-Whitehead
Date:       June 9, 2023
Description: Generates motion for an object.
"""

import numpy as np
from pyquaternion import Quaternion
import math
from tqdm import tqdm
import re
import os
parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('object_name', help='Name of the object')
args = parser.parse_args()


object_name = args.object_name
proj_dir : str = os.path.dirname(os.path.dirname(__file__))
input_output_directory : str = os.path.join(proj_dir,"input")
output_dir : str = os.path.join(proj_dir,"output")

################################################
# Object properties

# Initial position [m] [x,y,z]
p0 = np.array([40, 0, 90])
# Initial attitude quaternion [q0,qx,qy,qz]
rotation_angle = np.pi / 2  # 90 degrees in radians
rotation_axis = np.array([0, 1, 0])  # rotation around the y-axis
q0 = Quaternion(axis=rotation_axis, angle=rotation_angle)
# Initial velocity vector [m/s] [x,y,z]
v0 = np.array([-3.5, 0, -5])
# Initial rotation vector [rad/s] [0,x,y,z]
w0 = np.array([math.pi/8, 0, 0])

#Inertia matrix of the object [kg*m^2]


"""np.array([
        [31878.03,   0.00,   0.00],
        [  0.00, 190743.17,   0.00],
        [  0.00,   0.00, 184580.73]
    ])"""

################################################
# Simulation properties

# Simulation time step [s]
dt = 0.001
# Simulation duration [s]
sim_t = 15
# Time between frames [s]
frame_t = 0.1

################################################
def read_diagM_file(file_path):
    with open(file_path, 'r') as f:
        # Read the entire file content
        file_content = f.read()

        # Find the numpy array string using a regular expression
        if res := re.search(r'np.array\(\[([\s\S]*?)\]\)', file_content):
            matrix_string = res[1]
        else:
            raise ValueError("No inertia matrix found in file.")

        # Remove square brackets and spaces, then convert the matrix string into a list of lists of floats
        matrix = np.array([list(map(float, re.sub(r'[\[\]\s]', '', line).split(','))) for line in matrix_string.strip().split(',\n') if line.strip()])
        return matrix

I = read_diagM_file(os.path.join(input_output_directory,object_name + "_diagonalized_inertia_matrix"))
# Main program

def main():
    P, Q = create_object_motion(p0, q0, v0, w0, dt, sim_t, frame_t, I)
    print(P, Q)
    

def create_object_motion(p0 : np.ndarray, q0 : Quaternion, v0 : np.ndarray, w0 : np.ndarray,
                  dt : float, sim_t : float, frame_t : float, 
                  I : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """This function creates a motion for the given initial conditions.

    Args:
        p0 (np.ndarray) [x, y, z]: Initial position of the object
        q0 (Quaternion) [qw, qx, qy, qz]: Initial orientation of the object
        v0 (np.ndarray) [x, y, z]: Initial velocity of the object
        w0 (np.ndarray) [x, y, z]: Initial angular velocity of the object
        dt (float): Time step for calculations
        sim_t (float): Duration of simulation
        frame_t (float): Time step between each sample
        I (np.ndarray): Inertia matrix of the object

    Raises:
        ValueError: If there is a discrepancy in the energy during the simulation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Positions and orientations at each sampled time.
    """
    
    w0 = np.insert(w0, 0, 0) # Insert a zero at the beginning of the rotation vector (for quaternion multiplication)
    
    # Create tumble motion
    positions, orientations, _, T = f15_tumble_integrator(p0, q0, v0, w0, dt, sim_t, frame_t, I)

    # Check the percentage change in total energy
    initial_energy = T[0]
    final_energy = T[-1]
    percentage_change = abs(final_energy - initial_energy) / initial_energy * 100

    if percentage_change > 0.01:
        raise ValueError("Energy dissipated by more than 0.01% during the simulation. Please consider reducing the simulation time step.")
    
    # Convert list of Quaternion objects to 2D NumPy array
    orientations = np.vstack([quat.elements for quat in orientations])
    return positions, orientations
    


def quatDE(x : Quaternion):
    return np.array(
        [
            [x[0], -x[1], -x[2], -x[3]],
            [x[1], x[0], -x[3], x[2]],
            [x[2], x[3], x[0], -x[1]],
            [x[3], -x[2], x[1], x[0]],
        ]
    )

def f15_tumble_integrator(p : np.ndarray, q : Quaternion, v : np.ndarray, 
                          w : np.ndarray, dt : float, sim_t : float, frame_t : float, 
                          I : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Inputs:
      `p` - initial position vector (body-inertial) [m] [x,y,z]
      `q` - initial attitude quaternion [q0,qx,qy,qz]
      `v` - initial velocity vector (body-inertial) [m/s] [x,y,z]
      `w` - initial rotation vector (body-inertial) [rad/s] [0,x,y,z]
      `dt` - simulation time step [s]. Recommend a small time step for stability (ex 0.01s)
      `sim_t` - simulation duration [s]
      `frame_t` - time between frames [s]
      `I` - inertia matrix of the object [kg*m^2]

    Output:
      `P` - position time history
      `Q` - quaternion time history
      `W` - rotation vector time history
      `T` - total energy time history
    '''

    # Check normalization
    if math.isclose(sum(q.elements**2), 1, abs_tol=1e-5) == False:
        print('Warning: Initial quaternion is not normalized')
        print(f'sum(qi^2) = {str(sum(q.elements**2))}')

    # Simulation parameters
    t = np.arange(0, sim_t, dt)
    N = len(t)

    # Calculate frame step for saving data
    frame_step = int(frame_t/dt)

    # Calculate the number of output frames based on frame_t
    num_output_frames = int(sim_t / frame_t)

    # Preallocate Output with the correct size
    T = np.zeros(N) # Total energy time history (at each time step)
    # All other outputs are saved at every frame_step
    W = np.zeros((num_output_frames, 3))
    Q = np.array([Quaternion(0, 0, 0, 0)] * num_output_frames)
    P = np.zeros((num_output_frames, 3))

    # Create a counter for the output frames
    output_frame_idx = 0

    ## 4th Order Runge Kutta Integration Simulation (RK4)

    # Euler's equations of motion
    def f(ww, I):
        return -np.linalg.inv(I) @ np.cross(ww, I @ ww)

    # Initialize k vectors (slope estimators)
    k1 = np.zeros(3)
    k2 = np.zeros(3)
    k3 = np.zeros(3)
    k4 = np.zeros(3)

    for i in tqdm(range(N)):
        # (1) Update position
        p = p + v*dt

        # (2) Update attitude
        qR = quatDE(q)
        q_dot = 0.5 * np.dot(qR, w)
        q = q + Quaternion(q_dot*dt)

        q = q / np.sum(q.elements**2) # Enforce quaternion constraint: sum(qi^2) = 1

        # (3) Runge Kutta Update Acceleration
        w_ = w[1:]  # rotation rate x,y,z [rad/s]

        k1 = f(w_, I)
        k2 = f(w_ + (dt/2)*k1, I)
        k3 = f(w_ + (dt/2)*k2, I)
        k4 = f(w_ + dt*k3, I)

        w_ = w_ + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        # (4) Calculate energy state
        T[i] = 0.5 * np.dot(w_, I @ w_)

        # (5) Update rotation vector
        w[1:] = w_

        # Save data at every frame_step
        if i % frame_step == 0:
            P[output_frame_idx] = p
            Q[output_frame_idx] = q
            W[output_frame_idx, :] = w_

            # Increment the output frame counter
            output_frame_idx += 1
    return P, Q, W, T
    
if __name__ == "__main__":
    main()