# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

__all__ = ['BUILDER', 'MAP_FUNC']

BUILDER = Registry('builder', scope='mmengine')
MAP_FUNC = Registry('map_fn', scope='mmengine')
