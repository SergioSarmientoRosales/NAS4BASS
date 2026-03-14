from __future__ import annotations


def select_surrogate_models(
    models: list,
    model_names: list[str],
    selected_names: list[str] | None = None,
):
    """
    If selected_names is None, return all models.
    Otherwise return only the models whose names match.
    """
    if selected_names is None:
        return models, model_names

    selected_names_set = set(selected_names)

    filtered_models = []
    filtered_names = []

    for model, name in zip(models, model_names):
        if name in selected_names_set:
            filtered_models.append(model)
            filtered_names.append(name)

    if not filtered_models:
        raise ValueError(
            f"No models matched selected_names={selected_names}"
        )

    return filtered_models, filtered_names