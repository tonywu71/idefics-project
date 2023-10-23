import pytest
from pathlib import Path
from scripts.infer_on_idefics import main as infer_on_idefics


def test_idefics_inference():
    """Test inference on the IDEFICS model."""
    try:
        infer_on_idefics(idefics_config_path=Path("tests/configs/idefics_config_test.yaml"),
                         inference_config_path=Path("tests/configs/inference_config_test.yaml"),
                         prompts=["This is a test prompt."])
    except Exception as e:
        pytest.fail("Inference failed. The following exception was raised: " + str(e))
