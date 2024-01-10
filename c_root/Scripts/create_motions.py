"""
Description: Generates motion for an object.
"""
import sys
import os
import site
user_site_packages = site.getusersitepackages()
print (user_site_packages)
sys.path.append(user_site_packages)
import numpy as np

from pyquaternion import Quaternion
import math
from tqdm import tqdm

import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.save_info_to_files_utils import save_camera_info_to_file
import Utils.dataset_constants as dc
from Utils import file_tools
from Utils import motion_rep as mr





# Set PYTHONPATH


###############################################

# Output properties

# Name of the output directory
desired_output_directory = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "input"
)

# Object properties

if len(sys.argv) > 2:
    object_name = sys.argv[-2]
    tumble_id = sys.argv[-1]

# Object ID
object_id = object_name.split("_")[0]
# Tumble ID
inertia_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "objects",
    "diag",
    f"{object_name}_diagonalized_inertia_matrix.txt",
)


# Inertia matrix of the object

I = file_tools.read_inertia_matrix_from_file(inertia_path)
# Simulation properties

# Simulation time step [s]
dt = dc.dt
# Simulation duration [s]
sim_t = dc.default_sim_t
# Time between frames [s]
frame_t = dc.frame_t
info_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "objects",
    "inertia",
    f"{object_name}_info.txt",
)
# bbox of current object
maxc,minc = file_tools.extract_corners(info_path)

# size of current object

size = np.abs(np.array(maxc) - np.array(minc))
effective_size = np.mean(size)

# min and max object-camera distance, based on  the size of the object
def inital_direction(origin,fov):
   #get the direction vector in a uniform distribution
    z = np.random.uniform(min_distance, max_distance)
    half_base_angle = np.deg2rad(fov / 2)
    half_base_size = z * np.tan(half_base_angle)
    x = np.random.uniform(-half_base_size, half_base_size)
    y = np.random.uniform(-half_base_size, half_base_size)
    direction = np.array([x,y,z])
    return direction

def calculate_distance_with_fov(
    effective_size, fov_degrees, image_dimension, coverage_ratio
):
    """
    Calculates the distance from the camera to the object based on the desired coverage ratio of the image,
    assuming a typical camera field of view.
    """
    fov_radians = math.radians(fov_degrees)
    covered_image_dimension = coverage_ratio * image_dimension
    distance = (effective_size / 2) / math.tan(fov_radians / 2)
    adjusted_distance = distance * (image_dimension / covered_image_dimension)
    return adjusted_distance


# min object-camera distance
min_distance = calculate_distance_with_fov(
    effective_size, dc.camera_fov, 1024, dc.min_coverage_ratio
)

# max object-camera distance
max_distance = calculate_distance_with_fov(
    effective_size, dc.camera_fov, 1024, dc.max_coverage_ratio
)

# camera field of view
fov = dc.camera_fov

#############################################
# Sun orientation

# Select random sun orientation in cartesian coordinates
sun_orientation: np.ndarray | None = None  # None for random orientation

def get_random_light_energy():
    # Define the possible sun energy values
    possible_energies = dc.possible_light_energies

    # Randomly select one intensity value
    return random.choice(possible_energies)


################################################


def create_object_motion(
    p0: np.ndarray,
    q0: Quaternion,
    v0: np.ndarray,
    w0: np.ndarray,
    dt: float,
    sim_t: float,
    frame_t: float,
    I: np.ndarray,
    max_dist: float,
    min_dist: float,
    fov: int,
) -> tuple[np.ndarray, np.ndarray]:
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

    w0 = np.insert(
        w0, 0, 0
    )  # Insert a zero at the beginning of the rotation vector (for quaternion multiplication)

    # Create tumble motion
    all_positions, orientations, _, T = tumble_integrator(
        p0, q0, v0, w0, dt, sim_t, frame_t, I
    )

    # Check the percentage change in total energy
    initial_energy = T[0]
    final_energy = T[-1]
    percentage_change = abs(final_energy - initial_energy) / initial_energy * 100

    if percentage_change > 0.01:
        raise ValueError(
            "Energy dissipated by more than 0.01% during the simulation. Please consider reducing the simulation time step."
        )

    # Filter positions and orientations to only those within the frustum
    frustum_positions = []
    frustum_orientations = []

    for pos, orient in zip(all_positions, orientations):
        if is_in_frustum(pos, max_dist, min_dist, fov):
            frustum_positions.append(pos)
            frustum_orientations.append(orient.elements)

    # Convert list of positions and orientations to 2D NumPy arrays
    frustum_positions = np.array(frustum_positions)
    frustum_orientations = np.array(frustum_orientations)

    return frustum_positions, frustum_orientations

    # Checks if a given point is in the frustum


def is_in_frustum(
    point: np.ndarray, max_dist: float, min_dist: float, fov: int
) -> bool:
    half_base_angle = np.deg2rad(fov / 2)
    half_base_size = max_dist * np.tan(half_base_angle)

    # Translate point back to origin
    x, y, z = point

    # Check if point is within the minimum and maximum distance
    if z < min_dist or z > max_dist:
        return False

    # Calculate the size of the frustum at the height of the point
    point_height = z
    frustum_height = max_dist
    frustum_ratio = point_height / frustum_height
    half_point_base_size = half_base_size * frustum_ratio

    # Check if the point is inside the frustum's base at its height
    return (
        -half_point_base_size <= x <= half_point_base_size
        and -half_point_base_size <= y <= half_point_base_size
    )

    # Generates a random sun vector


def generate_random_unit_vector() -> np.ndarray:
    phi = 2 * np.pi * np.random.random()  # azimuthal angle
    theta = np.arccos(2 * np.random.random() - 1)  # polar angle

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array([x, y, z])

    # Generates an initial position in the frustum

def line_plane_intersection(point, direction, plane_normal, plane_point):
    point = np.array(point)
    direction = np.array(direction)
    plane_normal = np.array(plane_normal)
    plane_point = np.array(plane_point)

    # Calculate the dot product of the direction vector and the plane normal
    dot_product = np.dot(direction, plane_normal)

    # Check if the line is parallel to the plane
    if abs(dot_product) < 1e-6:
        return None  # No intersection, the line is parallel to the plane

    # Calculate the parameter t for the line-plane intersection
    t = np.dot(plane_point - point, plane_normal) / dot_product

    # Calculate the intersection point
    intersection_point = point + t * direction

    return t
def generate_optimal_initial_conditions(max_dist, min_dist, fov):
    # Generate a random position within a more central area of the frustum
    def generate_initial_position():
        # Define a factor to limit the range for x and y (e.g., 0.5 for half the frustum width)
        z = np.random.uniform(
            min_dist, max_dist
        )
       
        half_base = z * np.tan(np.deg2rad(fov / 2)) 
        x = np.random.uniform(-half_base, half_base)
        y = np.random.uniform(-half_base, half_base)
        
        return np.array([x, y, z])

    # Generate a random, normalized quaternion
    def generate_initial_quaternion():
        q = Quaternion.random()
        return q

    # Generate a moderate initial velocity vector
   

    # Generate a moderate initial rotation vector
    def generate_initial_rotation():
        w = np.random.uniform(-1, 1, 3)
        return w

    p0 = generate_initial_position()
    def generate_initial_velocity():
        side1_norm = [0,0,1]
        side2_norm = [0,0,-1]
        side3_norm = [math.cos(math.radians(fov/2)),0,math.sin(math.radians(fov/2))]
        side4_norm = [-math.cos(math.radians(fov/2)),0,math.sin(math.radians(fov/2))]
        side5_norm = [0,math.cos(math.radians(fov/2)),math.sin(math.radians(fov/2))]
        side6_norm = [0,-math.cos(math.radians(fov/2)),math.sin(math.radians(fov/2))]
        direction  = np.random.uniform(-1, 1, 3)
        direction = direction/np.linalg.norm(direction)
              
        t1 = line_plane_intersection(p0, direction, side1_norm, [0,0,min_dist])
        t2 = line_plane_intersection(p0, direction, side2_norm, [0,0,max_dist])
        t3 = line_plane_intersection(p0, direction, side3_norm, [0,0,0])
        t4 = line_plane_intersection(p0, direction, side4_norm, [0,0,0])
        t5 = line_plane_intersection(p0, direction, side5_norm, [0,0,0])
        t6 = line_plane_intersection(p0, direction, side6_norm, [0,0,0])
        t = [t1,t2,t3,t4,t5,t6]
        t = [i for i in t if i is not None and i > 0]
        t = min(t)
        p = p0 + t*direction
        print(f"p: {p}")
        
        
        direction = p - p0
        # we want at least dc.min_frames of the object to be in the frustum
        # so we want the object to move at least dc.min_frames*dc.frame_t
        print(f"direction: {direction}")
        print(f"t: {t}")
        print(f"dc.min_frames*dc.frame_t: {dc.min_frames*dc.frame_t}")
        direction = direction/(dc.min_frames*dc.frame_t)
        print(f"direction: {direction}")
        print(f"tumble_id: {tumble_id}")
        return direction
       #get the point where the direction vector intersects the frustum
    
       
        
        
    q0 = generate_initial_quaternion()
    v0 = generate_initial_velocity()
    w0 = generate_initial_rotation()

    return p0, q0, v0, w0

    # Initial position, attitude quaternion, velocity and angular velocity vectors


p0, q0, v0, w0 = generate_optimal_initial_conditions(max_distance, min_distance, fov)


# Select random sun orientation in cartesian coordinates
if sun_orientation is None:
    sun_orientation = generate_random_unit_vector()


def quatDE(x: Quaternion):
    return np.array(
        [
            [x[0], -x[1], -x[2], -x[3]],
            [x[1], x[0], -x[3], x[2]],
            [x[2], x[3], x[0], -x[1]],
            [x[3], -x[2], x[1], x[0]],
        ]
    )


def tumble_integrator(
    p: np.ndarray,
    q: Quaternion,
    v: np.ndarray,
    w: np.ndarray,
    dt: float,
    sim_t: float,
    frame_t: float,
    I: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
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
    """

    # Check normalization
    if math.isclose(sum(q.elements**2), 1, abs_tol=1e-5) == False:
        print("Warning: Initial quaternion is not normalized")
        print(f"sum(qi^2) = {str(sum(q.elements**2))}")

    # Simulation parameters
    t = np.arange(0, sim_t, dt)
    N = len(t)

    # Calculate frame step for saving data
    frame_step = int(frame_t / dt)

    # Calculate the number of output frames based on frame_t
    num_output_frames = int(sim_t / frame_t)

    # Preallocate Output with the correct size
    T = np.zeros(N)  # Total energy time history (at each time step)
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
        p = p + v * dt

        # (2) Update attitude
        qR = quatDE(q)
        q_dot = 0.5 * np.dot(qR, w)
        q = q + Quaternion(q_dot * dt)

        q = q / np.sum(q.elements**2)  # Enforce quaternion constraint: sum(qi^2) = 1

        # (3) Runge Kutta Update Acceleration
        w_ = w[1:]  # rotation rate x,y,z [rad/s]

        k1 = f(w_, I)
        k2 = f(w_ + (dt / 2) * k1, I)
        k3 = f(w_ + (dt / 2) * k2, I)
        k4 = f(w_ + dt * k3, I)

        w_ = w_ + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

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
#a function that returns the initial direction of the tumble
# that chooses this direction based on the position of the object
# and with a higher probability of choosing a directions that points towards the (0,0,0) point

    
    
   
    
 
    
    

    

def main(output_directory: str, object_id: str):
    # Ensure output_directory exists
    output_path = os.path.join(output_directory, f"{object_id}_{tumble_id}")
    os.makedirs(output_path, exist_ok=True)

    # Creates the positions (P) and quaternions (Q) for an object
    P, Q = create_object_motion(
        p0, q0, v0, w0, dt, sim_t, frame_t, I, max_distance, min_distance, fov
    )
    print(f"Number of frames: {len(P)}")
    # Now let's write the positions (P) and quaternions (Q) to a file
    scene_gt_path = f"{output_path}/scene_gt.txt"  # Construct the full path to the file
    with open(scene_gt_path, "w") as file:
        for frame_number, (pos, quat) in enumerate(zip(P, Q)):
            # Assuming pos is a numpy array [x, y, z] and quat is a numpy array [qw, qx, qy, qz]
            formatted_line = f"{frame_number:05d},{object_id},{quat[1]:.6f},{quat[2]:.6f},{quat[3]:.6f},{quat[0]:.6f},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}\n"
            file.write(formatted_line)

    def write_scene_gt_info(
        output_path: str,
        object_group_id: str,
        sim_t: float,
        dt: float,
        frame_t: float,
        P0: np.ndarray,
        Q0: Quaternion,
        V0: np.ndarray,
        W0: np.ndarray,
    ):
        scene_gt_info_path = f"{output_path}/scene_gt_info.txt"
        with open(scene_gt_info_path, "w") as file:
            file.write(f"----- Tumble {object_group_id} info -----\n\n")
            file.write(f"Object Group ID: {object_group_id}\n\n")
            file.write("--- Simulation info ---\n")
            file.write(f"Max duration of simulation: {sim_t:.3f} s\n")
            file.write(f"Simulation duration: {frame_t*len(P):.1f} s\n")
            file.write(f"Time between frames: {frame_t:.1f} s\n")
            file.write(f"Number of frames: {len(P)}\n\n")

            file.write(f"--- Lighting info ---\n")
            file.write(
                f"Sun orientation [x, y, z] (unit vector): {np.array2string(sun_orientation, separator=', ', precision=6)}\n"
            )
            file.write("--- Object {object_group_id} info ---\n\n")
            file.write(
                f"Initial position [x, y, z] (m): [{P0[0]:.1f}, {P0[1]:.1f}, {P0[2]:.1f}]\n"
            )
            file.write(
                f"Initial attitude quaternion [qw, qx, qy, qz]: [{Q0[0]:.1f}, {Q0[1]:.1f}, {Q0[2]:.1f}, {Q0[3]:.1f}]\n"
            )
            file.write(
                f"Initial velocity vector [x, y, z] (m/s): [{V0[0]:.1f}, {V0[1]:.1f}, {V0[2]:.1f}]\n"
            )
            file.write(
                f"Initial angular velocity vector [x, y, z] (rad/s): [{W0[0]:.6f}, {W0[1]:.6f}, {W0[2]:.6f}]\n"
            )
            file.write(f"Min/Max distance: [{min_distance}, {max_distance}]\n")

            # Open the file for writing sun energy values
            sun_energy_path = f"{output_path}/sun_energy.txt"
            with open(sun_energy_path, 'w') as sun_energy_file:
                for frame_number, (pos, quat) in enumerate(zip(P, Q)):
                    sun_energy = get_random_light_energy()
                    # Write the frame number and corresponding sun energy to the file
                    sun_energy_file.write(f"{frame_number:05d},{sun_energy}\n")
    # Write camera information to file
    save_camera_info_to_file(output_path)
    mr.show_motion(P,object_name, output_directory, tumble_id)

    # Write scene_gt_info
    write_scene_gt_info(
        output_path,
        object_group_id=object_id,
        sim_t=sim_t,
        dt=dt,
        frame_t=frame_t,
        P0=p0,
        Q0=q0,
        V0=v0,
        W0=w0,
    )


if __name__ == "__main__":
    main(desired_output_directory, object_id)
