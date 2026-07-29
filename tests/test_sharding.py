"""The sharded core, on 8 simulated CPU devices. Phase 1.

    XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest tests/

Sharding logic, PartitionSpec errors, shard_map signatures and collective placement all
reproduce faithfully on CPU. Do not rent a GPU to debug a sharding annotation.

    test_device_invariance          same seed on 1 device and on 8 simulated devices gives
                                    the same update. rtol=1e-12 in f32, near bitwise.
                                    The most important test in the repo.
    test_strategy_A_equals_B        the two contraction strategies produce the same update
                                    for the same seed
    test_comm_volume_A              instrument collectives, assert A moves O(N) not O(Nd)
    test_comm_volume_B              assert B moves exactly one params-sized psum per
                                    generation
    test_state_sharding             distribution state carries the intended NamedSharding,
                                    not replicated

What simulated devices do not model is interconnect bandwidth or latency. Every correctness
claim is answerable here; no timing claim is.
"""
