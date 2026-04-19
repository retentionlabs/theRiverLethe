from transformers.modeling_utils import *

from ...ops.self_retention import eager_linear_forward, eager_compiled_linear_forward, is_eager_supported_kernel
from ...integrations.flash_linear_attention import flash_linear_attention_forward, is_fla_available, is_fla_supported_kernel


class RetentionInterface(GeneralInterface):
    """
    Dict-like object keeping track of allowed retention functions. You can easily add a new retention function
    with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
    it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {
        "auto": (lambda *args, **kwargs: None, lambda *args, **kwargs: None),
        "flash_linear_attention": (flash_linear_attention_forward, is_fla_supported_kernel),
        "eager|compiled": (eager_linear_forward, is_eager_supported_kernel),
        "eager": (eager_linear_forward, is_eager_supported_kernel)
    }

    def get_interface(self, config, rettn_implementation: str, default: Callable) -> Callable:
        """Return the requested `attn_implementation`. Also, strictly check its validity, and raise if invalid."""
        if rettn_implementation is None:
            logger.warning_once(
                "You tried to access the `RetentionInterface` with a `config.rettn_implementation` set to `None`. This "
                "is expected if you use an Attention Module as a standalone Module. If this is not the case, something went "
                "wrong with the dispatch of `config._rettn_implementation`"
            )
        elif rettn_implementation not in self:
            raise KeyError(
                f"`{rettn_implementation}` is not a valid retention implementation registered in the `RetentionInterface`"
            )

        if rettn_implementation == "auto":
            # TODO: Add auto select logic
            pass

        impl, support_checker = super().get(rettn_implementation, default)
        is_supported, detailed_impl_name = support_checker(config.memory_shape, config.memory_depth, config.memory_loss_type)
        if not is_supported:
            raise KeyError(
                f"`{rettn_implementation}` is not a valid retention implementation for `{detailed_impl_name}` registered in the `RetentionInterface`"
            )
        return impl


AdaptationInterface = RetentionInterface

ALL_RETENTION_FUNCTIONS: RetentionInterface = RetentionInterface()
ALL_ADAPTATION_FUNCTIONS = ALL_RETENTION_FUNCTIONS
