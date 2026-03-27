export PYTHONPATH=$(cd "$dir" && pwd):${PYTHONPATH}

pytest tests/accuracy/operators/test_activation.py
pytest tests/accuracy/operators/test_attention.py
pytest tests/accuracy/operators/test_convolution.py
pytest tests/accuracy/operators/test_embedding.py
pytest tests/accuracy/operators/test_gemm.py
pytest tests/accuracy/operators/test_kv_cache.py
pytest tests/accuracy/operators/test_linear.py
pytest tests/accuracy/operators/test_normalization.py
pytest tests/accuracy/operators/test_position_embedding.py
pytest tests/accuracy/operators/test_sampling.py
pytest tests/accuracy/operators/test_store_lowrank.py