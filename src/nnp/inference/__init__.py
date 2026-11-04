"""Inference: evaluation folder → results.txt + time.txt (see docs/modules/inference)."""

from nnp.inference.predictor import InferenceRun, infer_folder, run, write_outputs

__all__ = ["InferenceRun", "infer_folder", "run", "write_outputs"]
