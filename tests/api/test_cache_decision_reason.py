import torch

from cache_dit import DBCacheConfig
from cache_dit.caching.cache_contexts.cache_context import CachedContext
from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager


def make_manager(**config_kwargs):
    cache_config_kwargs = {
        "max_warmup_steps": 0,
        "enable_separate_cfg": False,
    }
    cache_config_kwargs.update(config_kwargs)
    context = CachedContext(
        cache_config=DBCacheConfig(**cache_config_kwargs)
    )
    manager = CachedContextManager()
    manager._current_context = context
    return manager


def test_can_cache_records_early_miss_reasons():
    states = torch.ones(1, 2, 4)

    warmup_manager = make_manager(max_warmup_steps=1)
    warmup_manager.current_context.executed_steps = 1
    assert not warmup_manager.can_cache(states)
    assert warmup_manager.get_last_cache_decision_reason() == "warmup"

    disabled_manager = make_manager(residual_diff_threshold=0.0)
    assert not disabled_manager.can_cache(states)
    assert disabled_manager.get_last_cache_decision_reason() == "threshold_disabled"

    empty_manager = make_manager(residual_diff_threshold=0.08)
    assert not empty_manager.can_cache(states)
    assert empty_manager.get_last_cache_decision_reason() == "no_previous_buffer"


def test_can_cache_records_similarity_outcomes():
    states = torch.ones(1, 2, 4)
    manager = make_manager(residual_diff_threshold=0.08)
    manager.set_Fn_buffer(states.clone())

    assert manager.can_cache(states)
    assert manager.get_last_cache_decision_reason() == "similarity_hit"

    assert not manager.can_cache(torch.zeros_like(states))
    assert manager.get_last_cache_decision_reason() == "similarity_threshold"


def test_similarity_records_shape_mismatch():
    manager = make_manager(residual_diff_threshold=0.08)

    assert not manager.similarity(
        torch.ones(1, 2, 4),
        torch.ones(1, 3, 4),
        threshold=0.08,
    )
    assert manager.get_last_cache_decision_reason() == "shape_mismatch"
