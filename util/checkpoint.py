from collections import OrderedDict


def normalize_state_dict_keys(state_dict):
    """
    Normalize legacy checkpoint keys to the cleaned DFECrack module names.

    This keeps previously trained weights loadable after repository cleanup.
    """
    replacements = (
        ('.MFS.', '.segmentation_head.'),
        ('.GBC_C.', '.local_context.'),
    )

    normalized = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('MFS.'):
            new_key = 'segmentation_head.' + new_key[len('MFS.'):]
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        normalized[new_key] = value
    return normalized
