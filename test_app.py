import pytest
from PIL import Image
import numpy as np
from app import parse_class_name, overlay_cam

#TEXT_PARSER UNIT TESTS
def test_parse_class_name_regex_format():
    raw = "a image of Peach leaves diseased by Bacterial spot with symptoms"
    res = parse_class_name(raw)
    
    assert res["plant"] == "Peach"
    assert res["condition"] == "Bacterial Spot"
    assert res["is_healthy"] is False
    assert res["short"] == "Peach — Bacterial Spot"

def test_parse_class_name_underscore_format():
    raw = "Tomato_Early_blight"
    res = parse_class_name(raw)
    
    assert res["plant"] == "Tomato"
    assert res["condition"] == "Early Blight"
    assert res["is_healthy"] is False

def test_parse_class_name_healthy():
    raw = "a photo of Apple leaves healthy"
    res = parse_class_name(raw)
    
    assert res["is_healthy"] is True
    assert res["condition"] == "Healthy"


#GRADCAM UNIT TESTS
def test_overlay_cam_output_shape_and_range():
    pil_img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
    mock_cam = np.random.rand(14, 14)
    output = overlay_cam(pil_img, mock_cam, size=336)
    
    assert output.shape == (336, 336, 3)
    assert output.min() >= 0.0
    assert output.max() <= 1.0


#STREAMLIT APP TEST
from streamlit.testing.v1 import AppTest

def test_streamlit_initial_state():
    at = AppTest.from_file("app.py").run()
    
    assert at.info[0].value == "Upload photo to start the analysis."
    assert at.toggle[0].value is True  # show_top3 True by default
    assert at.toggle[1].value is False # show_gradcam False by default