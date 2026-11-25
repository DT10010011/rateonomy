import os
from collections.abc import Mapping

from PIL import Image


LABELED_IMAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "image.png"
)
_LABELED_IMAGE_KEYS = {
    "annotated_image",
    "labeled_image",
    "visualisation",
    "visualization",
}


def _find_image(value):
    if isinstance(value, Image.Image):
        return value

    if isinstance(value, Mapping):
        for nested_value in value.values():
            image = _find_image(nested_value)
            if image is not None:
                return image
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            image = _find_image(nested_value)
            if image is not None:
                return image

    return None


def _find_labeled_image(value):
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if str(key).casefold() in _LABELED_IMAGE_KEYS:
                image = _find_image(nested_value)
                if image is not None:
                    return image
        for nested_value in value.values():
            image = _find_labeled_image(nested_value)
            if image is not None:
                return image
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            image = _find_labeled_image(nested_value)
            if image is not None:
                return image

    return None


def save_and_open_labeled_image(workflow_result):
    image = _find_labeled_image(workflow_result)
    if image is None:
        raise RuntimeError(
            "Roboflow did not return a labeled image. Ensure the workflow "
            "exposes its visualization output."
        )

    image.save(LABELED_IMAGE_PATH)
    os.startfile(LABELED_IMAGE_PATH)
    return LABELED_IMAGE_PATH
