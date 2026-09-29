import json
from pathlib import Path

import numpy as np
import pytest

from src.event_delta.json_io import atomic_write_json, json_dumps, to_jsonable


def test_nested_numpy_scalars_and_path_preserve_json_types(tmp_path):
    value = {
        "nested": [{"flag": np.bool_(True), "count": np.int64(7)},
                   {"f32": np.float32(1.25), "f64": np.float64(2.5)}],
        "path": tmp_path / "result.json",
    }
    destination = tmp_path / "nested.json"
    atomic_write_json(destination, value)
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == {"nested": [{"flag": True, "count": 7}, {"f32": 1.25, "f64": 2.5}],
                      "path": str(tmp_path / "result.json")}
    assert isinstance(loaded["nested"][0]["flag"], bool)
    assert isinstance(loaded["nested"][0]["count"], int)
    assert isinstance(loaded["nested"][1]["f32"], float)


def test_unknown_or_non_scalar_values_report_field_path_and_do_not_replace_file(tmp_path):
    destination = tmp_path / "config.json"
    destination.write_text('{"previous": true}\n', encoding="utf-8")
    with pytest.raises(TypeError, match=r"\$\.results\.feature.*shape=\(2, 3\)"):
        atomic_write_json(destination, {"results": {"feature": np.zeros((2, 3))}})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"previous": True}


def test_json_dumps_rejects_unknown_objects_with_nested_path():
    with pytest.raises(TypeError, match=r"unsupported JSON value at \$\.items\[0\]\.value"):
        json_dumps({"items": [{"value": object()}]})


def test_torch_scalar_device_and_dtype_when_torch_is_available():
    torch = pytest.importorskip("torch")
    normalized = to_jsonable({"scalar": torch.tensor(np.int64(9)),
                              "device": torch.device("cpu"), "dtype": torch.float32})
    assert normalized == {"scalar": 9, "device": "cpu", "dtype": "torch.float32"}
    assert isinstance(normalized["scalar"], int)
    with pytest.raises(TypeError, match=r"non-scalar torch tensor at \$\.tensor"):
        to_jsonable({"tensor": torch.zeros(2)})
