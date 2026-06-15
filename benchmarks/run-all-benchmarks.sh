#!/usr/bin/env bash
# Kept for backwards-compatibility. The real entrypoint is ../run.sh (host-tagged
# results, storage benchmark, auto-build). This just forwards to it.
exec "$(dirname "$0")/../run.sh" "$@"
