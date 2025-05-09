#!/usr/bin/env python3

import habitat_sim
import habitat_sim.agent
import habitat_sim.utils.common as utils
import numpy as np
import quaternion
import imageio
import os
import shutil
from math import ceil

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
    rgb_sensor.resolution = [1280, 960]
    rgb_sensor.position = [0.0, 1.5, 0.0]

    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth_sensor"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [1280, 960]
    depth_sensor.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


sim = habitat_sim.Simulator(make_cfg())

pathfinder = sim.pathfinder
assert pathfinder.is_loaded, (
    f"Navmesh not loaded - check the config at {scene_config_file}"
)
seed = 424242
pathfinder.seed(seed)

start = pathfinder.get_random_navigable_point()
goal = pathfinder.get_random_navigable_point()

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
ty = interpolated[0][1]

# Walk path and observe #########################################################
rgb_timestamps = []
depth_timestamps = []
positions = []


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
    positions.append(f"{i:03d} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")


agent = sim.get_agent(0)
prev_rotation = None
prev_pos = None
rot_threshold = 0.02
extra_steps = 0
for i, pos in enumerate(interpolated):
    pos = pos.copy()
    pos[1] = ty
    state = habitat_sim.AgentState()
    frame = i + extra_steps

    if i + 1 < len(interpolated):
        forward = np.array(interpolated[i + 1]) - np.array(pos)
        forward[1] = 0  # flatten to straight ahead
        forward = forward / np.linalg.norm(forward)
        rotation = utils.quat_from_two_vectors(np.array([0, 0, -1]), forward)

        if prev_rotation:
            rot_diff = np.quaternion(
                rotation.w - prev_rotation.w, 0, rotation.y - prev_rotation.y, 0
            )
        else:
            rot_diff = np.quaternion(0, 0, 0, 0)

        # Sharp turn, interpolate further
        if (abs(rot_diff.w) + abs(rot_diff.y)) > rot_threshold:
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

print(
    f"Rendered {len(interpolated) + extra_steps} frames from {start} to {goal}, with seed {seed}"
)
sim.close()
