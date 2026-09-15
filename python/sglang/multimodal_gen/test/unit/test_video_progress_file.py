import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "runtime/utils/video_progress.py"
spec = importlib.util.spec_from_file_location("video_progress_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
JOB = "813514c3-4443-4552-bee0-7b2b5a31d940"


class VideoProgressFileTests(unittest.TestCase):
    def test_atomic_job_scoped_progress_has_no_tensors(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SGLANG_VIDEO_PROGRESS_DIR": tmp}):
            for step in (0, 1, 17, 50):
                module.write_video_progress(JOB, step, 50, "denoising")
                value = json.loads((Path(tmp) / f"{JOB}.json").read_text())
                self.assertEqual(value["step"], step)
                self.assertEqual(value["job_id"], JOB)
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
            module.write_video_progress("../invalid", 1, 50, "denoising")
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)

    def test_disabled_reporting_does_nothing(self):
        with patch.dict(os.environ, {"SGLANG_VIDEO_PROGRESS_DIR": ""}), patch.object(Path, "mkdir") as mkdir:
            module.write_video_progress(JOB, 1, 50, "denoising")
            mkdir.assert_not_called()

    def test_telemetry_failure_does_not_fail_generation(self):
        with patch.dict(os.environ, {"SGLANG_VIDEO_PROGRESS_DIR": "/tmp/not-used"}), patch.object(Path, "mkdir", side_effect=OSError("test")):
            module.write_video_progress(JOB, 1, 50, "denoising")


if __name__ == "__main__":
    unittest.main()
