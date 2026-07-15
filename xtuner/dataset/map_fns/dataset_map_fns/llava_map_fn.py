# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.registry import MAP_FUNC
from xtuner.utils import DEFAULT_IMAGE_TOKEN


@MAP_FUNC.register_module('llava_image_only_map_fn')
def llava_image_only_map_fn(example):
    # input contains the DEFAULT_IMAGE_TOKEN only
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            assert DEFAULT_IMAGE_TOKEN in msg['value']
            input += DEFAULT_IMAGE_TOKEN
        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    result = {'conversation': conversation}
    if 'route_family' in example:
        result['route_family'] = example['route_family']
    return result


@MAP_FUNC.register_module('llava_map_fn')
def llava_map_fn(example):
    """Unified mapper for both classification and regression tasks.
    Handles regression when 'category' field contains 'regression'.
    """
    messages = example['conversations']
    input = ''
    conversation = []
    
    # Skip initial gpt messages
    while messages and messages[0]['from'] == 'gpt':
        messages = messages[1:]
    
    # Process conversation messages
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                msg['value'] = DEFAULT_IMAGE_TOKEN + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']
        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    
    # Start with mandatory field
    result = {'conversation': conversation}

    # Pass through optional supervision targets / metadata so that later
    # stages (tokenize, collate) can still access them, especially when
    # we disable packing for eval.
    for key in [
        'survival_targets',      # dict with target_y / at_risk_mask
        'regression_targets',    # scalar or list
        'category', 'route_family', 'id', 'image', 'conversations', 'wsi_features',
        'division', 'project', 'image_len'
    ]:
        if key in example:
            result[key] = example[key]

    return result


@MAP_FUNC.register_module('llava_text_only_map_fn')
def llava_text_only_map_fn(example):
    """Text-only variant of llava_map_fn.

    It keeps the same supervision and metadata fields but removes visual
    placeholders and disables visual loading downstream.
    """
    messages = example['conversations']
    input_text = ''
    conversation = []

    while messages and messages[0]['from'] == 'gpt':
        messages = messages[1:]

    for msg in messages:
        if msg['from'] == 'human':
            value = msg['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            input_text += value
        elif msg['from'] == 'gpt':
            conversation.append({'input': input_text, 'output': msg['value']})
            input_text = ''
        else:
            raise NotImplementedError

    result = {
        'conversation': conversation,
        'image': None,
        'wsi_features': None,
        'image_len': 0,
    }
    for key in ['survival_targets', 'regression_targets', 'category', 'route_family', 'id', 'project']:
        if key in example:
            result[key] = example[key]

    return result
