import inspect


def _patch_vllm_init_app_state() -> None:
    try:
        from vllm.entrypoints.openai import api_server
    except Exception:
        return

    original = getattr(api_server, "init_app_state", None)
    if original is None:
        return

    try:
        param_count = len(inspect.signature(original).parameters)
    except Exception:
        return

    if getattr(api_server, "_seedmitigating_init_app_state_patched", False):
        return

    async def _compat_init_app_state(*args, **kwargs):
        if len(args) >= 4:
            return await original(*args, **kwargs)

        if len(args) != 3:
            return await original(*args, **kwargs)

        engine_client, state, parsed_args = args

        vllm_config = getattr(state, "vllm_config", None)

        if vllm_config is None:
            getter = getattr(engine_client, "get_vllm_config", None)
            if getter is not None:
                try:
                    value = getter()
                    vllm_config = await value if inspect.isawaitable(value) else value
                except Exception:
                    vllm_config = None

        if vllm_config is None:
            vllm_config = getattr(engine_client, "vllm_config", None)

        if vllm_config is None:
            raise RuntimeError("Cannot infer vllm_config for init_app_state compatibility shim")

        return await original(engine_client, vllm_config, state, parsed_args)

    if param_count == 4:
        api_server.init_app_state = _compat_init_app_state
        api_server._seedmitigating_init_app_state_patched = True


_patch_vllm_init_app_state()
