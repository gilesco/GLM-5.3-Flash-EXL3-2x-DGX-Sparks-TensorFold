# Copy to scripts/local.sh (not in git) and fill in this setup's values; scripts/config.sh reads it first.
WORKER=giles@192.168.2.2          # ssh target of the second Spark (key-based: ssh-copy-id user@192.0.2.2)
PORT=8000
# FABRIC_PEER=192.0.2.12       # the worker's CX7 address, when WORKER is reached over another network

# --- TensorFold v0.6.4 experiment (branch tensorfold-v0.6.4) ---
# The patches on this branch are rebased onto TensorFold v0.6.4 (was v0.6.0).
TF_VERSION=v0.6.4
# Own kernel cache so the experiment never shares compiled kernels with the production v0.6.0 image.
KERNEL_CACHE=$HOME/.cache/tensorfold-glm53-v064
