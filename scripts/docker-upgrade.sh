#!/usr/bin/env sh
set -eu

LOCAL_DOCKER_IMAGE="reforcexy-freqtrade"
REMOTE_DOCKER_IMAGE="freqtradeorg/freqtrade:stable_freqairl"

echo_timestamped() {
  command echo "$(date +"%Y-%m-%d %H:%M:%S") - $*"
}

if ! command -v docker >/dev/null 2>&1; then
  echo_timestamped "Error: docker not found in PATH"
  exit 1
fi

if [ ! -f "docker-compose.yml" ] && [ ! -f "docker-compose.yaml" ]; then
  echo_timestamped "Error: docker-compose.yml or docker-compose.yaml not found in current directory"
  exit 1
fi

echo_timestamped "Info: docker image pull for ${REMOTE_DOCKER_IMAGE}"
local_digest=$(command docker image inspect --format='{{.Id}}' "$REMOTE_DOCKER_IMAGE" 2>/dev/null || command echo "none")
if ! command docker image pull --quiet "$REMOTE_DOCKER_IMAGE"; then
  echo_timestamped "Error: docker image pull failed for ${REMOTE_DOCKER_IMAGE}"
  exit 1
fi
remote_digest=$(command docker image inspect --format='{{.Id}}' "$REMOTE_DOCKER_IMAGE" 2>/dev/null || command echo "none")

if [ "$local_digest" != "$remote_digest" ]; then
  echo_timestamped "Info: docker image ${REMOTE_DOCKER_IMAGE} was updated ($local_digest -> $remote_digest), wait for reload..."
  echo_timestamped "Info: restarting docker image ${REMOTE_DOCKER_IMAGE}"
  command docker compose --progress quiet down
  command docker image rm "$LOCAL_DOCKER_IMAGE"
  command docker compose --progress quiet up -d
  echo_timestamped "Info: restarted docker image ${REMOTE_DOCKER_IMAGE}"
else
  echo_timestamped "Info: docker image ${REMOTE_DOCKER_IMAGE} is up to date"
fi
