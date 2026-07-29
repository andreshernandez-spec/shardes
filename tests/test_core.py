"""ask / eval / tell end to end, single device.

    test_ask_tell_roundtrip   sphere and Rastrigin: ES actually descends, for every strategy
    test_state_is_pure        init/ask/tell return new state, nothing mutates in place
    test_two_line_diff        the Qiu and EGGROLL configurations both construct and run

Phase 1. Sharding lives in test_sharding.py.
"""
