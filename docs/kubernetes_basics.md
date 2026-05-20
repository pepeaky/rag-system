# Kubernetes Basics

## What is Kubernetes?
Kubernetes (K8s) is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. Originally developed by Google, it is now maintained by the Cloud Native Computing Foundation (CNCF).

## Core Concepts

### Pods
A Pod is the smallest deployable unit in Kubernetes. It represents a single instance of a running process and can contain one or more containers that share storage and network resources.

### Deployments
A Deployment provides declarative updates for Pods and ReplicaSets. You describe a desired state in a Deployment, and the Deployment Controller changes the actual state to the desired state at a controlled rate.

### Services
A Service is an abstraction that defines a logical set of Pods and a policy by which to access them. Services enable loose coupling between dependent Pods. Types include ClusterIP, NodePort, LoadBalancer, and ExternalName.

### ConfigMaps and Secrets
ConfigMaps allow you to decouple configuration from container images. Secrets are similar but designed for sensitive data like passwords and API keys. Both can be consumed as environment variables or mounted as volumes.

### Namespaces
Namespaces provide a mechanism for isolating groups of resources within a single cluster. They are intended for use in environments with many users spread across multiple teams or projects.

## Architecture

### Control Plane
- **kube-apiserver**: Frontend for the Kubernetes control plane
- **etcd**: Consistent and highly-available key-value store for cluster data
- **kube-scheduler**: Watches for newly created Pods with no assigned node and selects a node for them
- **kube-controller-manager**: Runs controller processes (Node, Job, EndpointSlice, ServiceAccount)

### Worker Nodes
- **kubelet**: Agent that runs on each node, ensures containers are running in a Pod
- **kube-proxy**: Network proxy that maintains network rules on nodes
- **Container runtime**: Software responsible for running containers (containerd, CRI-O)
