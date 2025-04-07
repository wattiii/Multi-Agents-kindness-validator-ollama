#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=$(pwd)
pytest tests