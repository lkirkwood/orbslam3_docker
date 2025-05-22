#!/usr/bin/env sh

habsim_traj=$(mktemp /tmp/XXXX)
sed 's/ /,/g' /Datasets/habsim-Replica/positions > $habsim_traj

orbslam_traj=$(mktemp /tmp/XXXX)
sed 's/ /,/g' /ORB_SLAM3/CameraTrajectory.txt > $orbslam_traj

evo_traj euroc -p $habsim_traj $orbslam_traj
