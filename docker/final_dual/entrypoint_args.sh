#!/usr/bin/env bash

final_dual_parse_entrypoint_args() {
  local first_arg="${1:-}"
  case "$#:$first_arg" in
    2:*)
      TEST_FILE="$1"
      RESULT_DIR="$2"
      ;;
    3:/root/participant/start.sh)
      TEST_FILE="$2"
      RESULT_DIR="$3"
      ;;
    4:bash)
      if [[ "$2" != "/root/participant/start.sh" ]]; then
        echo "ERROR: only /root/participant/start.sh is an allowed command" >&2
        return 2
      fi
      TEST_FILE="$3"
      RESULT_DIR="$4"
      ;;
    *)
      echo "Usage: docker run IMAGE TEST_FILE RESULT_DIR" >&2
      echo "   or: docker run IMAGE bash /root/participant/start.sh TEST_FILE RESULT_DIR" >&2
      echo "   or: docker run IMAGE /root/participant/start.sh TEST_FILE RESULT_DIR" >&2
      return 2
      ;;
  esac
}
