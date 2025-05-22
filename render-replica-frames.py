#!/usr/bin/env python3

import habitat_sim
import habitat_sim.agent
import habitat_sim.utils.common as utils
import numpy as np
import quaternion
import imageio
import os
import shutil

scene_config_file = "/Datasets/Replica/scene_dataset_config.json"
output_dir = "/Datasets/habsim-Replica"
shutil.rmtree(output_dir + "/rgb")
os.makedirs(output_dir + "/rgb", exist_ok=True)
shutil.rmtree(output_dir + "/depth")
os.makedirs(output_dir + "/depth", exist_ok=True)


def make_cfg() -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_config_file
    sim_cfg.gpu_device_id = -1

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "rgb_sensor"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [960, 1280]
    rgb_sensor.position = [0.0, 1.5, 0.0]

    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth_sensor"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [960, 1280]
    depth_sensor.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def interpolate_path(start, goal) -> list:
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = goal
    found = pathfinder.find_path(path)

    if not found:
        print("No path found.")
        sim.close()
        exit(1)

    step_size = 0.025
    interpolated = []
    for i in range(len(path.points) - 1):
        p0 = np.array(path.points[i])
        p1 = np.array(path.points[i + 1])
        segment = p1 - p0
        dist = np.linalg.norm(segment)
        num_steps = int(np.floor(dist / step_size))
        if num_steps == 0:
            interpolated.append(p0.tolist())
            continue
        direction = segment / dist
        for j in range(num_steps):
            interpolated.append(p0 + j * step_size * direction)
    interpolated.append(path.points[-1])
    return interpolated


sim = habitat_sim.Simulator(make_cfg())

pathfinder = sim.pathfinder
assert pathfinder.is_loaded, (
    f"Navmesh not loaded - check the config at {scene_config_file}"
)
seed = 424242
pathfinder.seed(seed)

waypoints = [
    [2.7400379180908203, -1.600136637687683, -3.5185205936431885],
    [6.584749698638916, -1.600136637687683, -3.7201988697052],
    [6.034043312072754, -1.600136637687683, -5.86156702041626],
    [3.4886186599731445, -1.600136637687683, -5.4488091468811035],
]

path_points = []
for wp_num in range(len(waypoints)):
    start = waypoints[wp_num]
    goal = waypoints[wp_num + 1] if (wp_num + 1) < len(waypoints) else waypoints[0]
    path_points.extend(interpolate_path(start, goal))

ty = -1.6

# Walk path and observe #########################################################
rgb_timestamps = []
depth_timestamps = []
positions = []

TIMESCALE = 10**9


def observe(frame: int):
    obs = sim.get_sensor_observations()

    cur_time = frame / 30.0
    timestamp = f"{cur_time:03f}"
    rgb_frame = obs["rgb_sensor"]
    rgb_frame_path = f"rgb/frame_{timestamp}.png"
    imageio.imwrite(f"{output_dir}/{rgb_frame_path}", rgb_frame)

    depth_frame = obs["depth_sensor"]
    depth_frame_path = f"depth/frame_{timestamp}.exr"
    imageio.imwrite(f"{output_dir}/{depth_frame_path}", depth_frame)

    rgb_timestamps.append(f"{timestamp}\n")
    depth_timestamps.append(
        f"{timestamp} {rgb_frame_path} {timestamp} {depth_frame_path}\n"
    )

    tx, _, tz = pos
    qx, qy, qz, qw = (
        state.rotation.x,
        state.rotation.y,
        state.rotation.z,
        state.rotation.w,
    )
    positions.append(f"{cur_time * TIMESCALE:03f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")


agent = sim.get_agent(0)
prev_rotation = None
prev_pos = None
rot_threshold = 0.02
extra_steps = 0
for i, pos in enumerate(path_points):
    pos = pos.copy()
    pos[1] = ty
    state = habitat_sim.AgentState()
    frame = i + extra_steps

    if i + 1 < len(path_points):
        forward = np.array(path_points[i + 1]) - np.array(pos)
        forward[1] = 0  # flatten to straight ahead

        if all(forward == [0.0, 0.0, 0.0]):
            continue
        forward /= np.linalg.norm(forward)
        print(f"frame {i} - {int(i / len(path_points) * 100)}%", end="\r")

        rotation = utils.quat_from_two_vectors(
            np.array([0, 0, -1]), forward
        ).normalized()

        if prev_rotation:
            rot_diff = np.quaternion(
                rotation.w - prev_rotation.w, 0, rotation.y - prev_rotation.y, 0
            )
        else:
            rot_diff = np.quaternion(0, 0, 0, 0)

        # Sharp turn, interpolate further
        if (abs(rot_diff.w) + abs(rot_diff.y)) > rot_threshold:
            print()
            print("=" * 80)
            print(f"Rotated sharply: {prev_rotation} -> {rotation}")
            print("=" * 80)
            rot_steps = int(max(abs(rot_diff.w), abs(rot_diff.y)) / rot_threshold)
            rot_step_delta = rot_diff / rot_steps
            pos_step_delta = (pos - prev_pos) / rot_steps
            extra_steps += rot_steps + 1
            for step_num in range(rot_steps + 1):
                # interpolate
                interp_rot = quaternion.slerp_evaluate(
                    prev_rotation, rotation, step_num / rot_steps
                )
                # move
                state.position = pos + (pos_step_delta * step_num)
                state.rotation = interp_rot
                # observe
                agent.set_state(state)
                frame += 1
                observe(frame)

        # No sharp turn, just move and observe
        else:
            state.position = pos
            state.rotation = rotation
            agent.set_state(state)
            observe(frame)

        prev_rotation = rotation
        prev_pos = pos


with open(f"{output_dir}/rgb-timestamps", "w") as stream:
    stream.writelines(rgb_timestamps)

with open(f"{output_dir}/depth-timestamps", "w") as stream:
    stream.writelines(depth_timestamps)

with open(f"{output_dir}/positions", "w") as stream:
    stream.writelines(positions)

print()
print(f"Rendered {len(path_points) + extra_steps} frames, with seed {seed}")
sim.close()
