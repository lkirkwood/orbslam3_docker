# checking if you have nvidia
if nvidia-smi | grep -q "Driver" 2>/dev/null; then
  echo "******************************"
  echo """It looks like you have nvidia drivers running. Please make sure your nvidia-docker is setup by following the instructions linked in the README and then run build_container_cuda.sh instead."""
  echo "******************************"
  while true; do
    read -p "Do you still wish to continue?" yn
    case $yn in
      [Yy]* ) make install; break;;
      [Nn]* ) exit;;
      * ) echo "Please answer yes or no.";;
    esac
  done
fi

runtime=docker
command -v podman && runtime=podman

# UI permisions
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.$runtime.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

xhost +local:$runtime

$runtime pull docker.io/jahaniam/orbslam3:ubuntu20_noetic_cpu

# Remove existing container
$runtime rm -f orbslam3 &>/dev/null

# Create a new container
$runtime run -td --privileged --net=host --ipc=host \
    --name="orbslam3" \
    -e "DISPLAY=$DISPLAY" \
    -e "QT_X11_NO_MITSHM=1" \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -e "XAUTHORITY=$XAUTH" \
    -e ROS_IP=127.0.0.1 \
    --cap-add=SYS_PTRACE \
    -v /etc/group:/etc/group:ro \
    -v `pwd`/ORB_SLAM3:/ORB_SLAM3 \
    -v `pwd`/Replica:/Replica \
    -v `pwd`/Datasets:/Datasets \
    docker.io/jahaniam/orbslam3:ubuntu20_noetic_cpu bash

# Compile ORB_SLAM3
echo "============================"
echo "|    Compiling orbslam3    |"
echo "============================"
$runtime exec -it orbslam3 bash -i -c "cd /ORB_SLAM3 && chmod +x build.sh && ./build.sh "

# Compile Replica
echo "==============================="
echo "|    Compiling replica sdk    |"
echo "==============================="
$runtime exec -it orbslam3 bash -i -c "cd /Replica && chmod +x build.sh && ./build.sh "
