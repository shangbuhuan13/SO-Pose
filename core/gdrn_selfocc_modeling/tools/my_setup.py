import logging


def setup_for_distributed(is_master):
    """This function disables printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print
    if not is_master:
        logging.getLogger("core").setLevel("WARN")
        logging.getLogger("d2").setLevel("WARN")
        logging.getLogger("lib").setLevel("WARN")
        logging.getLogger("my").setLevel("WARN")

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
