#!/usr/bin/env sh
set -eu

FREQTRADE_CONFIG="user_data/config.json"
LOCAL_DOCKER_IMAGE="reforcexy-freqtrade"
REMOTE_DOCKER_IMAGE="freqtradeorg/freqtrade:stable_freqairl"

echo_timestamped() {
  printf '%s - %s\n' "$(date +"%Y-%m-%d %H:%M:%S")" "$*"
}

LOCK_TAG=$(printf '%s' "$LOCAL_DOCKER_IMAGE" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_')
LOCKFILE="/tmp/docker-upgrade.${LOCK_TAG}.lock"
if [ -f "$LOCKFILE" ]; then
  echo_timestamped "Error: already running for ${LOCAL_DOCKER_IMAGE}"
  exit 1
fi
trap 'rm -f "$LOCKFILE"' 0 HUP INT TERM
touch "$LOCKFILE"

jsonc_to_json() {
  awk '
    BEGIN{in_str=0;esc=0;in_block=0;have_prev=0}
    {
      line=$0; len=length(line); i=1; out="";
      sub(/\r$/, "", line)
      while(i<=len){
        c=substr(line,i,1);
        nextc = (i<len)?substr(line,i+1,1):"\n";
        if(in_block){
          if(c=="*" && nextc=="/"){ in_block=0; i+=2; }
          else { i++; }
          continue;
        }
        if(!in_str){
          if(c=="/" && nextc=="/"){ break; }
          if(c=="/" && nextc=="*"){ in_block=1; i+=2; continue; }
          if(c=="\""){ in_str=1; out=out c; i++; continue; }
          out=out c; i++;
        } else {
          out=out c;
          if(esc){ esc=0; }
          else if(c=="\\"){ esc=1; }
          else if(c=="\""){ in_str=0; }
          i++;
        }
      }
      sub(/[[:space:]]+$/, "", out)
      if (out ~ /^[[:space:]]*$/) next
      cur = out
      if (have_prev){
        sub(/,[[:space:]]*\}[[:space:]]*$/, "}", prev)
        sub(/,[[:space:]]*\][[:space:]]*$/, "]", prev)
        if (prev ~ /,[[:space:]]*$/ && cur ~ /^[[:space:]]*[}\]]/) {
          sub(/,[[:space:]]*$/, "", prev)
        }
        key = (cur ~ /^[[:space:]]*"[^"]+"[[:space:]]*:/)
        openval = (cur ~ /^[[:space:]]*[{[]/)
        strval = (cur ~ /^[[:space:]]*"/) && !(key)
        numval = (cur ~ /^[[:space:]]*-?[0-9]/)
        boolnull = (cur ~ /^[[:space:]]*(true|false|null)([[:space:]]|,|]|\}|$)/)
        prev_value_end = (prev ~ /[}\]][[:space:]]*$/) || (prev ~ /"[[:space:]]*$/) || (prev ~ /-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?[[:space:]]*$/) || (prev ~ /(true|false|null)[[:space:]]*$/)
        if (prev_value_end && (key || openval || strval || numval || boolnull)) {
          prev = prev ","
        }
        print prev
      }
      prev = cur
      have_prev=1
    }
    END {
      if (have_prev){
        sub(/,[[:space:]]*\}[[:space:]]*$/, "}", prev)
        sub(/,[[:space:]]*\][[:space:]]*$/, "]", prev)
        sub(/,[[:space:]]*$/, "", prev)
        print prev
      }
    }
  ' "$1" | jq -c '.'
}

escape_telegram_markdown() {
  printf '%s' "$1" | command sed \
    -e 's/\\/\\\\/g' \
    -e 's/[][(){}.*_~`>#\+=|.!-]/\\&/g'
}

send_telegram_message() {
    if [ -z "${FREQTRADE_CONFIG_JSON:-}" ]; then
      FREQTRADE_CONFIG_JSON=$(jsonc_to_json "$FREQTRADE_CONFIG" 2>/dev/null || echo "")
    fi
    printf '%s' "$FREQTRADE_CONFIG_JSON" | jq empty 2>/dev/null || { echo_timestamped "Error: invalid JSON configuration"; exit 1; }

    freqtrade_telegram_enabled=$(printf '%s' "$FREQTRADE_CONFIG_JSON" | jq -r '.telegram.enabled // "false"' 2>/dev/null || echo "false")
    if [ "$freqtrade_telegram_enabled" = "false" ]; then
      return 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo_timestamped "Error: curl not found, cannot send telegram message"
      return 1
    fi

    telegram_message=$(escape_telegram_markdown "$1")
    if [ -z "$telegram_message" ]; then
      echo_timestamped "Error: message variable is empty"
      return 1
    fi

    freqtrade_telegram_token=$(printf '%s' "$FREQTRADE_CONFIG_JSON" | jq -r '.telegram.token // ""' 2>/dev/null || echo "")
    freqtrade_telegram_chat_id=$(printf '%s' "$FREQTRADE_CONFIG_JSON" | jq -r '.telegram.chat_id // ""' 2>/dev/null || echo "")
    if [ -n "$freqtrade_telegram_token" ] && [ -n "$freqtrade_telegram_chat_id" ]; then
      curl_error=$(command curl -s -X POST \
        --data-urlencode "text=${telegram_message}" \
        --data-urlencode "parse_mode=MarkdownV2" \
        --data "chat_id=$freqtrade_telegram_chat_id" \
        "https://api.telegram.org/bot${freqtrade_telegram_token}/sendMessage" 2>&1 1>/dev/null)
      if [ $? -ne 0 ]; then
        echo_timestamped "Error: failed to send telegram message: $curl_error"
        return 1
      fi
    fi
}

if ! command -v jq >/dev/null 2>&1; then
  echo_timestamped "Error: jq not found in PATH"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo_timestamped "Error: docker not found in PATH"
  exit 1
fi

if [ ! -f "docker-compose.yml" ] && [ ! -f "docker-compose.yaml" ]; then
  echo_timestamped "Error: docker-compose.yml or docker-compose.yaml file not found in current directory"
  exit 1
fi

if [ ! -f "$FREQTRADE_CONFIG" ]; then
  echo_timestamped "Error: $FREQTRADE_CONFIG file not found from current directory"
  exit 1
fi

echo_timestamped "Info: docker image pull for ${REMOTE_DOCKER_IMAGE}"
local_digest=$(command docker image inspect --format='{{.Id}}' "$REMOTE_DOCKER_IMAGE" 2>/dev/null || command echo "none")
if ! command docker image pull --quiet "$REMOTE_DOCKER_IMAGE"; then
  echo_timestamped "Error: docker image pull failed for ${REMOTE_DOCKER_IMAGE}"
  exit 1
fi
remote_digest=$(command docker image inspect --format='{{.Id}}' "$REMOTE_DOCKER_IMAGE" 2>/dev/null || command echo "none")

rebuild_local_image=false
if [ "$local_digest" != "$remote_digest" ]; then
  rebuild_local_image=true
  message="docker image ${REMOTE_DOCKER_IMAGE} was updated ($local_digest -> $remote_digest)"
  echo_timestamped "Info: $message"
  send_telegram_message "$message"
else
  echo_timestamped "Info: docker image ${REMOTE_DOCKER_IMAGE} is up to date"
fi

if [ "$rebuild_local_image" = true ]; then
  message="rebuilding and restarting docker image ${LOCAL_DOCKER_IMAGE}"
  echo_timestamped "Info: $message"
  send_telegram_message "$message"
  if ! command docker compose --progress quiet down; then
    echo_timestamped "Error: docker compose down failed"
    exit 1
  fi
  if ! command docker image rm "$LOCAL_DOCKER_IMAGE"; then
    echo_timestamped "Warning: docker image rm failed for ${LOCAL_DOCKER_IMAGE}"
  fi
  if ! command docker compose --progress quiet up -d; then
    echo_timestamped "Error: docker compose up failed"
    exit 1
  fi
  message="rebuilt and restarted docker image ${LOCAL_DOCKER_IMAGE}"
  echo_timestamped "Info: $message"
  send_telegram_message "$message"
else
  echo_timestamped "Info: no rebuild and restart needed for docker image ${LOCAL_DOCKER_IMAGE}"
fi

exit 0
