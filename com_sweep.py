"""Run a KVComm sweep with one resident sender/receiver pair.

From the repository root, inside an allocated GPU job:
    conda activate kvcomm_cf
    python -u com_sweep.py

Quick check with two Qasper examples and two alphas:
    python -u com_sweep.py --test_tasks qasper --alphas 1.0 0.0 --limit 2

Uses the existing requirements.txt (Transformers 4.53.3) and models.py runtime.
The default models and 64,000-token input budget match com.py. When a dataset
allows truncation, the original evaluator keeps the sender's beginning and end
within the combined sender + receiver budget, excluding generated tokens.
The run_alpha_sweep.sbatch job launches this runner; com.py remains independent.
No resume, parallel evaluation, or automatic context reduction is performed.

Offline tests: python -m pytest -q tests/test_com_sweep.py
"""

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass, field
import datetime
import gc
from itertools import islice
import json
import logging
import math
from pathlib import Path
import re
import time
import traceback
from types import SimpleNamespace

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import set_seed
from tqdm import tqdm

from eval import CommunicationEvaluator
from layer_importance import calc_layer_importance, get_top_layers
from models import CVCommunicator


TEST_TASKS = ("countries", "tipsheets", "hotpotqa", "qasper", "musique",
              "multifieldqa_en", "twowikimqa")
CSV_FIELDS = ["test_task", "alpha", "communication_result"]


def get_evaluator(task):
    # Dataset imports initialize NLTK resources. Keep --help free of that I/O.
    from dataloader import get_evaluator as factory
    return factory(task)


@dataclass
class AlignConfig:
    device: str = "cuda:0"
    seed: int = 42
    snapshot_path: str = "snapshots"
    model_A: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_B: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_input_length: int = 64 * 1000
    top_layers: float = 0.7
    calib_size: int = 1
    mu: float = 0.5
    sigma: float = 10.0
    test_tasks: list[str] = field(default_factory=lambda: list(TEST_TASKS))
    alphas: list[float] = field(default_factory=lambda: [i / 10 for i in range(10, -1, -1)])
    limit: int = 0
    run_name: str = "kvcomm-sweep"
    results_csv: str = ""
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_tags: str = ""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, default in vars(AlignConfig()).items():
        if isinstance(default, bool):
            parser.add_argument(f"--{name}", action="store_true", default=default)
        elif isinstance(default, list):
            parser.add_argument(f"--{name}", nargs="+", type=type(default[0]), default=default)
        else:
            parser.add_argument(f"--{name}", type=type(default), default=default)
    return AlignConfig(**vars(parser.parse_args(argv)))


def validate_config(cfg):
    if not cfg.test_tasks or any(task not in TEST_TASKS for task in cfg.test_tasks):
        raise ValueError(f"test_tasks must be chosen from {TEST_TASKS}")
    if len(set(cfg.test_tasks)) != len(cfg.test_tasks):
        raise ValueError("test_tasks must not contain duplicates")
    if not cfg.alphas or any(not math.isfinite(a) or not 0 <= a <= 1 for a in cfg.alphas):
        raise ValueError("alphas must be finite values between 0 and 1")
    if len(set(cfg.alphas)) != len(cfg.alphas):
        raise ValueError("alphas must not contain duplicates")
    if not math.isfinite(cfg.top_layers) or not 0 < cfg.top_layers <= 1:
        raise ValueError("top_layers must be greater than 0 and at most 1")
    if not math.isfinite(cfg.mu) or not 0 <= cfg.mu <= 1:
        raise ValueError("mu must be between 0 and 1")
    if not math.isfinite(cfg.sigma) or cfg.sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    if cfg.max_input_length < 2 or cfg.calib_size < 1 or cfg.limit < 0:
        raise ValueError("max_input_length >= 2, calib_size >= 1, and limit >= 0 are required")
    if cfg.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ValueError("log_level must be DEBUG, INFO, WARNING, or ERROR")
    if cfg.use_wandb and not cfg.wandb_project:
        raise ValueError("wandb_project is required with use_wandb")
    if torch.device(cfg.device).type not in ("cpu", "cuda"):
        raise ValueError("This runner supports cpu or cuda devices")
    if cfg.results_csv:
        output = Path(cfg.results_csv)
        if output.resolve() == output.with_suffix(".jsonl").resolve():
            raise ValueError("CSV and JSONL diagnostics must use distinct paths; choose a .csv filename")


def receiver_layers(model):
    backbone = model.model
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    return backbone.layers


def clear_attention_tensors(cv):
    # The existing communicator only creates B_attn_weights during calibration.
    if hasattr(cv, "B_attn_weights"):
        cv.B_attn_weights.clear()
    for block in receiver_layers(cv.B):
        if hasattr(block.self_attn, "attn_inputs"):
            block.self_attn.attn_inputs = None


def release_memory(device):
    # Call after leaving exception handlers so traceback frames cannot hold KV.
    gc.collect()
    if torch.device(device).type == "cuda" and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


@contextmanager
def log_to(path, level, console=False):
    root = logging.getLogger()
    previous_level = root.level
    handlers = [logging.FileHandler(path, encoding="utf-8")]
    if console:
        handlers.append(logging.StreamHandler())
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)
    try:
        yield
    finally:
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(previous_level)


class ExperimentSession:
    """Own model weights; communicators borrow them only within a stage."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_A = self.model_B = self.tokenizer = None

    def __enter__(self):
        try:
            configs = [AutoConfig.from_pretrained(name) for name in
                       (self.cfg.model_A, self.cfg.model_B)]
            for config in configs:
                text_config = getattr(config, "text_config", config)
                if int(self.cfg.top_layers * text_config.num_hidden_layers) == 0:
                    raise ValueError("top_layers would select zero layers")
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_B)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.model_A = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_A, device_map={"": self.cfg.device},
                torch_dtype=torch.float16, attn_implementation="sdpa")
            self.model_B = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_B, device_map={"": self.cfg.device},
                torch_dtype=torch.float16, attn_implementation="sdpa")
            for model, name in ((self.model_A, self.cfg.model_A), (self.model_B, self.cfg.model_B)):
                model.eval()
                model.requires_grad_(False)
                model.name = name
            self.tokenizer = tokenizer
            logging.info("Loaded sender and receiver once; models remain resident across cases")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        self.model_A = self.model_B = self.tokenizer = None
        release_memory(self.cfg.device)

    @contextmanager
    def communicator(self, layers_list=None, trace=False):
        receiver = self.model_B
        # models.py installs temporary attention modules for calibration. Keep
        # the originals so repeated tasks reuse the resident weights and each
        # tracer sees the original supported attention type.
        saved = [(block, block.self_attn) for block in receiver_layers(receiver)]
        cv = None
        try:
            cv = CVCommunicator(
                self.model_A, receiver, 0, len(saved) - 1,
                layers_list=[-1] if layers_list is None else layers_list,
                top_layers=self.cfg.top_layers, apply_attn_tracer=trace).eval()
            yield cv
        finally:
            if cv is not None:
                clear_attention_tensors(cv)
            for block, attention in saved:
                # Also clean up a partially constructed calibration wrapper.
                if block.self_attn is not attention:
                    block.self_attn.attn_inputs = None
                block.self_attn = attention


class SweepCommunicationEvaluator(CommunicationEvaluator):
    def __init__(self, evaluator, tokenizer, use_wandb, max_input_length=64 * 1000):
        super().__init__(evaluator, tokenizer, use_wandb, max_input_length)
        self.truncation_count = 0
        self.items_evaluated = 0
        self.sample_index = None
        self.sample_id = None
        self.input_lengths = {}

    def truncate_input(self, input_ids_A, input_ids_B):
        a_len, b_len = input_ids_A.shape[-1], input_ids_B.shape[-1]
        self.input_lengths = {"sender_original": a_len, "receiver": b_len, "sender_used": a_len}
        # Reuse com.py's evaluator policy, including the dataset's opt-out.
        input_ids_A, input_ids_B = super().truncate_input(input_ids_A, input_ids_B)
        self.input_lengths["sender_used"] = input_ids_A.shape[-1]
        if input_ids_A.shape[-1] < a_len:
            self.truncation_count += 1
        logging.debug("sample=%s input_lengths=%s", self.sample_index, self.input_lengths)
        return input_ids_A, input_ids_B

    def inference(self, model, cv, item):
        input_ids_A, input_ids_B = self.prepare_input_ids(item, cv.A, cv.B)
        out_A = model(input_ids=input_ids_A, use_cache=True, return_dict=True, logits_to_keep=1)
        donor_cache = out_A.past_key_values
        del out_A
        output = cv.generate(input_ids_B, attention_mask=torch.ones_like(input_ids_B),
                             out_A_past_key_values=donor_cache, **self.generate_args)[0]
        return self.get_response(output, input_ids_B.shape[-1])

    def _test(self, model_A, cv, limit=None, do_calc_layer_importance=False):
        total = len(self.evaluator) if limit is None else min(limit, len(self.evaluator))
        if total == 0:
            raise ValueError("Cannot evaluate an empty dataset")
        self.items_evaluated = self.truncation_count = 0
        self.layer_importance_total.clear()
        progress = tqdm(islice(iter(self.evaluator), total), total=total,
                        desc="communication", disable=do_calc_layer_importance)
        try:
            for index, item in enumerate(progress, 1):
                self.sample_index = index
                self.sample_id = str(item.get("id", index))
                self.input_lengths = {}
                response = self.inference(model_A, cv, item)
                if do_calc_layer_importance:
                    try:
                        cv.calc_attn_weights_from_qk()
                        calc_layer_importance(cv.B_attn_weights, model_A.name,
                                              self.layer_importance_total)
                    finally:
                        clear_attention_tensors(cv)
                self.evaluator.evaluate_item(item, response)
                self.items_evaluated += 1
                progress.set_description(f"communication result: {self.evaluator.get_result():.4f}")
        finally:
            progress.close()
        return self.evaluator.get_result()


def gpu_memory(device, reset=False):
    if torch.device(device).type != "cuda" or not torch.cuda.is_initialized():
        return {"peak_allocated_gib": None, "peak_reserved_gib": None,
                "allocated_gib": None, "reserved_gib": None}
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    gib = 1024 ** 3
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
            "allocated_gib": torch.cuda.memory_allocated(device) / gib,
            "reserved_gib": torch.cuda.memory_reserved(device) / gib}


def append_record(path, record):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, allow_nan=False) + "\n")
        output.flush()


def run_stage(session, cfg, stage, log_path, journal, dataset=None, importance=None):
    """Contain recoverable errors here so their traceback dies before next case."""
    started = time.perf_counter()
    record = {"stage": stage, "test_task": cfg.test_task,
              "alpha": cfg.alpha if stage == "evaluation" else None,
              "status": "error", "communication_result": None,
              "configuration": vars(cfg).copy(), "selected_layers": [],
              "log_path": str(log_path), "max_input_length": cfg.max_input_length}
    evaluator = None
    summary = None
    with log_to(log_path, cfg.log_level):
        try:
            set_seed(cfg.seed)
            gpu_memory(cfg.device, reset=True)
            logging.info("Starting %s: task=%s alpha=%s cap=%s", stage, cfg.test_task,
                         record["alpha"], cfg.max_input_length)
            if dataset is None:
                dataset = get_evaluator(cfg.test_task)
            evaluator = SweepCommunicationEvaluator(dataset, session.tokenizer, False, cfg.max_input_length)
            trace = stage == "calibration"
            if not trace:
                get_top_layers(importance, cfg)
                record["selected_layers"] = [int(layer) for layer in cfg.layers_list]
            with session.communicator(layers_list=None if trace else cfg.layers_list, trace=trace) as cv:
                score = evaluator.test(session.model_A, cv,
                                       limit=cfg.calib_size if trace else (cfg.limit or None),
                                       do_calc_layer_importance=trace, no_wandb=True)
                if trace:
                    summary = {int(layer): [float(value) for value in values] for layer, values
                               in evaluator.layer_importance_total.items()}
                    record["layer_importance"] = summary
                else:
                    record["communication_result"] = float(score)
            record["status"] = "success"
        except torch.OutOfMemoryError as error:
            record.update(status="oom", error_type=type(error).__name__, error=str(error))
            # A buffering logging handler can retain exc_info and every tensor
            # in its traceback. Preserve the diagnostic as text only.
            logging.error("Case failed; its partial score will not be saved\n%s", traceback.format_exc())
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error))
            logging.exception("Unexpected error; stopping the suite")
            raise
        finally:
            record.update(elapsed_seconds=time.perf_counter() - started, **gpu_memory(cfg.device))
            record.update(items_evaluated=evaluator.items_evaluated if evaluator else 0,
                          truncation_count=evaluator.truncation_count if evaluator else 0,
                          sample_index=evaluator.sample_index if evaluator else None,
                          sample_id=evaluator.sample_id if evaluator else None,
                          input_lengths=evaluator.input_lengths.copy() if evaluator else {})
            record["configuration"] = vars(cfg).copy()
            record["configuration"]["layers_list"] = record["selected_layers"]
            append_record(journal, record)
            logging.info("Finished %s task=%s alpha=%s status=%s items=%s truncated=%s memory=%s",
                         stage, cfg.test_task, record["alpha"], record["status"],
                         record["items_evaluated"], record["truncation_count"],
                         record["peak_allocated_gib"])
    # Do not store exceptions or traceback objects in records or returned values.
    evaluator = None
    release_memory(cfg.device)
    return record, dataset, summary


def main(cfg):
    validate_config(cfg)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", cfg.run_name) or "kvcomm-sweep"
    suite_dir = Path(cfg.snapshot_path) / f"{name}-ctx{cfg.max_input_length}_{stamp}"
    suite_dir.mkdir(parents=True)
    results_csv = Path(cfg.results_csv) if cfg.results_csv else suite_dir / "results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    journal = results_csv.with_suffix(".jsonl")
    if results_csv.exists() and results_csv.stat().st_size:
        with results_csv.open(newline="") as output:
            if next(csv.reader(output), None) != CSV_FIELDS:
                raise ValueError(f"Unexpected CSV header in {results_csv}")
    metadata = dict(configuration=asdict(cfg), results_csv=str(results_csv), journal=str(journal),
                    torch_version=torch.__version__, transformers_version=transformers.__version__,
                    cuda_runtime=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(cfg.device) if torch.device(cfg.device).type == "cuda"
                    and torch.cuda.is_available() else None)
    (suite_dir / "configuration.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results = []
    with log_to(suite_dir / "suite.log", cfg.log_level, console=True):
        logging.info("Suite directory: %s; results: %s; diagnostics: %s", suite_dir, results_csv, journal)
        logging.info("Environment: %s", metadata)
        with ExperimentSession(cfg) as session:
            for task in cfg.test_tasks:
                case = SimpleNamespace(**asdict(cfg), test_task=task, alpha=1.0, layers_list=[-1])
                calibration, dataset, importance = run_stage(
                    session, case, "calibration", suite_dir / f"{task}-calibration.log", journal)
                try:
                    for index, alpha in enumerate(cfg.alphas, 1):
                        case = SimpleNamespace(**asdict(cfg), test_task=task, alpha=alpha, layers_list=[-1])
                        case_log = suite_dir / f"{task}-{index:02d}-alpha-{alpha:g}.log"
                        if calibration["status"] != "success":
                            record = {"stage": "evaluation", "test_task": task, "alpha": alpha,
                                      "status": "calibration_failed", "communication_result": None,
                                      "configuration": vars(case), "selected_layers": [],
                                      "max_input_length": cfg.max_input_length,
                                      "error": calibration.get("error", "Calibration failed"),
                                      "calibration_log": calibration["log_path"], "log_path": str(case_log),
                                      "items_evaluated": 0, "truncation_count": 0}
                            with log_to(case_log, cfg.log_level):
                                logging.error("Skipping task=%s alpha=%s: %s", task, alpha, record["error"])
                            append_record(journal, record)
                        else:
                            wandb_run = None
                            try:
                                if cfg.use_wandb:
                                    import wandb
                                    wandb_run = wandb.init(project=cfg.wandb_project,
                                                          entity=cfg.wandb_entity or None,
                                                          name=f"{suite_dir.name}-{task}-alpha-{alpha:g}",
                                                          tags=[t.strip() for t in cfg.wandb_tags.split(",") if t.strip()],
                                                          config=vars(case))
                                record, _, _ = run_stage(session, case, "evaluation", case_log,
                                                         journal, dataset, importance)
                                if record["status"] == "success":
                                    with results_csv.open("a", newline="", encoding="utf-8") as output:
                                        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
                                        if output.tell() == 0:
                                            writer.writeheader()
                                        writer.writerow({key: record[key] for key in CSV_FIELDS})
                                        output.flush()
                                if wandb_run is not None:
                                    wandb_run.log({"communication_result": record["communication_result"],
                                                   "status": record["status"],
                                                   "peak_allocated_gib": record["peak_allocated_gib"]})
                            finally:
                                if wandb_run is not None:
                                    wandb_run.finish()
                        results.append(record)
                finally:
                    dataset = importance = None
                    release_memory(cfg.device)
        succeeded = sum(record["status"] == "success" for record in results)
        logging.info("Suite finished: %s/%s successful. CSV=%s JSONL=%s",
                     succeeded, len(results), results_csv, journal)
    return results


if __name__ == "__main__":
    main(parse_args())
