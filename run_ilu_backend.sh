export PYTHONPATH=$(cd "$dir" && pwd):${PYTHONPATH}

export MOJO_BACKEND=ixformer
# export MOJO_BACKEND=ttx

pytest tests/accuracy/operators/test_ilu_backend.py