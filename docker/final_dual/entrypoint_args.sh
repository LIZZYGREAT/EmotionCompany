#!/usr/bin/env bash

final_dual_parse_entrypoint_args() {
  ENTRYPOINT_IDLE=false
  ENTRYPOINT_START_SH=false
  ENTRYPOINT_SHELL_COMMAND=""
  TEST_FILE=""
  RESULT_DIR=""

  if [[ "$#" -eq 0 ]]; then
    ENTRYPOINT_IDLE=true
    return 0
  fi

  # Official direct form: docker run IMAGE TEST_FILE RESULT_DIR.
  # Only interpret this as the two official paths when the first argument
  # actually looks like the input file.  Anything else (bash -c ...,
  # /bin/bash ... , sh -c ... , a single command string, etc.) is passed
  # through unchanged instead of being rejected.
  if [[ "$#" -eq 2 && "$1" != -* && "$2" != -* ]]; then
    if [[ "$1" == *.jsonl || -f "$1" ]]; then
      TEST_FILE="$1"
      RESULT_DIR="$2"
      ENTRYPOINT_START_SH=true
      return 0
    fi
  fi

  # Some runners pass the whole command as one string.  Execute those through
  # a shell; a single bare executable/path is passed through unchanged.
  if [[ "$#" -eq 1 ]]; then
    if [[ "$1" == *" "* || "$1" == *";"* || "$1" == *"|"* || "$1" == *"&"* ]]; then
      ENTRYPOINT_SHELL_COMMAND="$1"
    fi
    return 0
  fi

  return 0
}
