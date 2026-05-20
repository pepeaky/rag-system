# Docker Networking

## Network Drivers

Docker uses a pluggable networking subsystem. Several drivers exist:

### Bridge
The default network driver. Bridge networks are usually used when applications run in standalone containers that need to communicate. Containers on the same bridge network can communicate, while isolation is maintained from containers not connected to that bridge.

### Host
Removes network isolation between the container and the Docker host. The container shares the host's networking namespace. This is useful when the container needs to handle a large range of ports.

### Overlay
Overlay networks connect multiple Docker daemons together and enable Swarm services to communicate with each other. They remove the need to do OS-level routing between containers on different hosts.

### Macvlan
Macvlan networks allow you to assign a MAC address to a container, making it appear as a physical device on your network. The Docker daemon routes traffic to containers by their MAC addresses.

## DNS Resolution
Docker provides automatic DNS resolution for containers. Containers on user-defined networks can resolve each other by container name. The embedded DNS server forwards external DNS queries to the host's DNS configuration.

## Port Publishing
By default, containers are isolated from the host network. To make a container accessible from outside, you publish ports using the -p flag: `docker run -p 8080:80 nginx`. This maps host port 8080 to container port 80.

## Network Security
Docker networks provide isolation by default. Containers can only communicate with other containers on the same network. For production environments, consider using encrypted overlay networks and avoiding the host network driver.
