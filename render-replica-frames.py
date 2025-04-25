#!/usr/bin/env python3

import habitat_sim
import habitat_sim.agent
import habitat_sim.utils.common as utils
import numpy as np
import imageio
import os

scene_config_file = "/Datasets/Replica/scene_dataset_config.json"
output_dir = "/Datasets/habsim-Replica"
os.makedirs(output_dir + "/rgb", exist_ok=True)


def make_cfg() -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_config_file
    sim_cfg.gpu_device_id = -1

    sensor_specs = []
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [512, 384]
    sensor_spec.position = [0.0, 1.5, 0.0]
    sensor_specs.append(sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


sim = habitat_sim.Simulator(make_cfg())

pathfinder = sim.pathfinder
assert pathfinder.is_loaded, (
    f"Navmesh not loaded - check the config at {scene_config_file}"
)
seed = 123456789
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


step_size = 0.01
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
        interpolated.append((p0 + j * step_size * direction).tolist())
interpolated.append(path.points[-1])


timestamps = []
for i, pos in enumerate(interpolated):
    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = pos

    if i + 1 < len(interpolated):
        forward = np.array(interpolated[i + 1]) - np.array(pos)
        forward[1] = 0  # flatten to avoid tilting camera
        forward = forward / np.linalg.norm(forward)
        rotation = utils.quat_from_two_vectors(np.array([0, 0, -1]), forward)
        state.rotation = rotation
    agent.set_state(state)

    obs = sim.get_sensor_observations()
    frame = obs["color_sensor"]
    imageio.imwrite(f"{output_dir}/rgb/frame_{i:03d}.png", frame)
    timestamps.append(f"{i:03d}\n")

with open(f"{output_dir}/timestamps", "w") as stream:
    stream.writelines(timestamps)

print(f"Rendered {len(interpolated)} frames from {start} to {goal}, with seed {seed}")
sim.close()
