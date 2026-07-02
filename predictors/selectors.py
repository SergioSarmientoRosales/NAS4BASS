from __future__ import annotations


def select_surrogate_models(
    models: list,
    model_names: list[str],
    selected_names: list[str] | None = None,
):
    """
    If selected_names is None, return all models.
    Otherwise return models in the exact order requested by selected_names.
    """
    if selected_names is None:
        return models, model_names

    duplicate_requested = sorted(
        {name for name in selected_names if selected_names.count(name) > 1}
    )
    if duplicate_requested:
        raise ValueError(
            f"Duplicate selected model names are not allowed: {duplicate_requested}"
        )

    duplicate_available = sorted(
        {name for name in model_names if model_names.count(name) > 1}
    )
    if duplicate_available:
        raise ValueError(
            f"Duplicate available model names are not supported: {duplicate_available}"
        )

    by_name = dict(zip(model_names, models))
    missing = [name for name in selected_names if name not in by_name]

    if missing:
        raise ValueError(
            f"Unknown selected model names: {missing}. "
            f"Available models: {sorted(model_names)}"
        )

    filtered_models = [by_name[name] for name in selected_names]
    filtered_names = list(selected_names)

    return filtered_models, filtered_names
