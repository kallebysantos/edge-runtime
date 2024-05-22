#!/usr/bin/env bash

# --features cli/tracing
ORT_DYLIB_PATH='./libonnxruntime.so' GIT_V_TAG=0.1.1 cargo build && SB_AI_MODELS_DIR='./temp/models/' EDGE_RUNTIME_PORT=9998 RUST_BACKTRACE=full ./target/debug/edge-runtime "$@" start \
	--main-service ./examples/main \
	--event-worker ./examples/event-manager
