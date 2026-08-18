#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=${MYLEETGPU_DATA_DIR:-"$repo_dir/data"}
username=${MYLEETGPU_LAN_USERNAME:-myleetgpu}
password_file="$data_dir/lan.htpasswd"

usage() {
  cat <<'EOF'
Usage: scripts/lan-auth.sh ensure|reset|address

  ensure   Keep existing credentials or create a strong password once.
  reset    Prompt for a replacement password (or use MYLEETGPU_LAN_PASSWORD).
  address  Print the primary non-loopback IPv4 address used for the LAN route.
EOF
}

detect_address() {
  local address
  address=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}')
  if [[ -z "$address" || "$address" == 127.* ]]; then
    echo "无法检测局域网 IPv4 地址；请设置 MYLEETGPU_LAN_ADDRESS。" >&2
    exit 1
  fi
  printf '%s\n' "$address"
}

write_credentials() {
  local password=${MYLEETGPU_LAN_PASSWORD:-}
  local generated=false

  if [[ -z "$password" && -t 0 ]]; then
    read -r -s -p "LAN 密码（至少 12 个字符，留空则随机生成）: " password
    printf '\n' >&2
  fi
  if [[ -z "$password" ]]; then
    password=$(openssl rand -base64 24 | tr -d '\n')
    generated=true
  fi
  if (( ${#password} < 12 )); then
    echo "LAN 密码至少需要 12 个字符。" >&2
    exit 1
  fi

  mkdir -p -- "$data_dir"
  umask 077
  printf '%s:%s\n' "$username" "$(openssl passwd -apr1 "$password")" >"$password_file"
  chmod 0644 "$password_file"

  echo "LAN 用户名: $username"
  if [[ "$generated" == true ]]; then
    echo "LAN 随机密码（仅显示这一次）: $password"
  else
    echo "LAN 密码已更新。"
  fi
}

case ${1:-} in
  ensure)
    if [[ -s "$password_file" ]]; then
      echo "沿用现有 LAN 凭据: $password_file"
    else
      write_credentials
    fi
    ;;
  reset)
    write_credentials
    ;;
  address)
    detect_address
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
