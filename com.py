import os
import torch
import argparse
import wandb
import datetime
import logging
import csv
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_utils import set_seed
from models_ac import ActivationCommunicator
from models import CVCommunicator
from models_cipher import CipherAgent
from typing import Literal
from utils import setup_logging, log_gpu_info, generate_run_name
from dataloader import get_evaluator
from eval import SkylineEvaluator, CommunicationEvaluator, BaselineEvaluator, ACEvaluator, NLDEvaluator, CipherEvaluator, clear_attention_tensors
from layer_importance import get_top_layers, get_layer_ranking, get_layer_scores, enumerate_layer_candidates
import random

RESULT_CSV_FIELDS = [
    "run_name", "calibration_method", "test_task", "model_A", "model_B", "seed",
    "top_layers", "calib_size", "limit", "alpha", "selected_layers",
    "calibration_f1", "communication_result", "snapshot_path",
]

@dataclass
class AlignConfig:
    # device configuration
    device: str = "cuda:0"
    seed: int = 42
    snapshot_path: str = "snapshots"
    # model configuration
    model_A: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_B: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_input_length: int = 64 * 1000
    # Communication configuration
    layer_from: int = 0
    layer_to: int = 26
    layers_list: list[int] = field(default_factory=lambda: [-1])
    top_layers: float = 0.0
    calib_size: int = 1
    breakpoint_calibration: bool = False
    always_include_layer_0: bool = True
    do_layer_curve: bool = False
    alpha: float = 1.0
    mu: float = 0.5
    sigma: float = 10.0
    random_selection: bool = False
    shift_back: bool = False
    # Test dataset configuration
    test_task: str = "tipsheets"
    task_name: str = ""
    limit: int = 0
    # Test configuration
    do_test: bool = False
    do_test_skyline: bool = False
    do_test_baseline: bool = False
    do_test_ac: bool = False
    do_test_nld: bool = False
    do_test_cipher: bool = False
    # NLD configuration
    # max tokens to generate for model A and B in phase 1
    nld_max_tokens_model_A_and_B_phase1: int = 128
    sender_aware: bool = False
    # AC configuration
    f: Literal["replace", "sum", "mean"] = "replace"
    layer_k: int = 26
    layer_j: int = 26
    # W&B configuration
    run_name: str = ""
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_tags: str = ""  # comma-separated tags
    # Logging configuration
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    results_csv: str = ""
    results_json: str = ""


@contextmanager
def communication_model(model_A, model_B, cfg, *, layers_list, trace=False):
    """Borrow the resident models and restore receiver attention after tracing."""
    backbone = model_B.model
    if hasattr(backbone, "language_model"):
        backbone = backbone.language_model
    saved = [(block, block.self_attn) for block in backbone.layers]
    cv = None
    try:
        cv = CVCommunicator(
            model_A, model_B, cfg.layer_from, cfg.layer_to,
            layers_list=layers_list, top_layers=cfg.top_layers,
            apply_attn_tracer=trace, shift_back=False if trace else cfg.shift_back,
        ).to(cfg.device).eval()
        yield cv
    finally:
        if cv is not None:
            clear_attention_tensors(cv)
        for block, attention in saved:
            if block.self_attn is not attention:
                block.self_attn.attn_inputs = None
            block.self_attn = attention


def calibrate_layer_sets(importance, communication_evaluator, model_A, model_B, cfg, snapshot_path):
    """Compare distinct sets on calib_size examples and retain the highest answer F1."""
    attention, prior = get_layer_scores(importance, cfg)
    candidates = enumerate_layer_candidates(
        attention, prior, int(cfg.top_layers * len(attention)),
        always_include_layer_0=cfg.always_include_layer_0)
    logging.info("Breakpoint calibration: %s distinct layer sets", len(candidates))
    winner = None
    candidate_results = []
    previous_scoring = communication_evaluator.score_f1
    try:
        communication_evaluator.score_f1 = True
        with (Path(snapshot_path) / "calibration_candidates.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=["alpha", "alpha_exact", "selected_layers", "f1"])
            writer.writeheader()
            for candidate_index, (midpoint, layers) in enumerate(candidates, start=1):
                set_seed(cfg.seed)
                with communication_model(model_A, model_B, cfg, layers_list=list(layers)) as cv:
                    score = float(communication_evaluator.test(model_A, cv, limit=cfg.calib_size, no_wandb=True))
                if not math.isfinite(score):
                    raise ValueError("Candidate F1 must be finite")
                record = dict(alpha=float(midpoint), alpha_exact=str(midpoint),
                              selected_layers=list(layers), f1=score)
                writer.writerow({**record, "selected_layers": json.dumps(record["selected_layers"])})
                output.flush()
                candidate_results.append(record)
                logging.info("Breakpoint candidate %d/%d: F1=%.6f | alpha=%.6f | layers=%s",
                             candidate_index, len(candidates), score, record["alpha"],
                             record["selected_layers"])
                # Preserve the first interval in an F1 tie.
                if winner is None or score > winner["f1"]:
                    winner = record
    finally:
        communication_evaluator.score_f1 = previous_scoring

    comparison = ["Calibration F1 comparison (highest first; %d examples per set):" % cfg.calib_size]
    for rank, record in enumerate(sorted(candidate_results, key=lambda item: item["f1"], reverse=True), start=1):
        comparison.append(
            f"  {rank}. F1={record['f1']:.6f} | alpha={record['alpha']:.6f}"
            f" | layers={record['selected_layers']}" + ("  <-- selected" if record is winner else ""))
    logging.info("\n%s", "\n".join(comparison))

    selection = {**winner, "candidate_count": len(candidates)}
    with (Path(snapshot_path) / "calibration_selection.json").open("w", encoding="utf-8") as output:
        json.dump(selection, output, indent=2, allow_nan=False)
        output.write("\n")
    cfg.alpha = winner["alpha"]
    cfg.layers_list = winner["selected_layers"]
    logging.info("Breakpoint selection: %s", selection)
    if cfg.use_wandb:
        wandb.log({"calibration_best_f1": winner["f1"], "calibration_best_alpha": cfg.alpha,
                   "calibration_selected_layers": cfg.layers_list})
    return winner["f1"]


def validate_result_files(cfg):
    if not cfg.results_csv and not cfg.results_json:
        return
    if not cfg.do_test or cfg.do_layer_curve:
        raise ValueError("Result exports require do_test without do_layer_curve")
    if cfg.results_csv and cfg.results_json and Path(cfg.results_csv).resolve() == Path(cfg.results_json).resolve():
        raise ValueError("CSV and JSON results must use distinct files")
    if cfg.results_csv:
        path = Path(cfg.results_csv)
        if path.exists() and path.stat().st_size:
            with path.open(newline="", encoding="utf-8") as output:
                if next(csv.reader(output), None) != RESULT_CSV_FIELDS:
                    raise ValueError(f"Unexpected CSV header in {path}")
    if cfg.results_json:
        path = Path(cfg.results_json)
        if path.exists() and path.stat().st_size:
            records = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
                raise ValueError(f"JSON results must contain a list of run records: {path}")


def save_communication_result(cfg, configuration, run_name, snapshot_path, layers, result, calibration_f1):
    """Append a completed final evaluation, keeping candidate F1 separate."""
    if not cfg.results_csv and not cfg.results_json:
        return
    method = "original"
    if cfg.breakpoint_calibration:
        method = "breakpoint"
    elif cfg.random_selection:
        method = "random"
    elif cfg.top_layers <= 0:
        method = "manual"
    record = dict(
        run_name=run_name, calibration_method=method, test_task=cfg.test_task,
        model_A=cfg.model_A, model_B=cfg.model_B, seed=cfg.seed,
        top_layers=cfg.top_layers, calib_size=cfg.calib_size, limit=configuration["limit"],
        alpha=float(cfg.alpha), selected_layers=[int(layer) for layer in layers],
        calibration_f1=calibration_f1, communication_result=float(result),
        snapshot_path=str(Path(snapshot_path).resolve()), configuration=configuration,
    )
    # Serialize before writing so unsupported or non-finite values cannot leave a partial row.
    json.dumps(record, allow_nan=False)
    if cfg.results_csv:
        path = Path(cfg.results_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=RESULT_CSV_FIELDS)
            if output.tell() == 0:
                writer.writeheader()
            row = {key: record[key] for key in RESULT_CSV_FIELDS}
            row["selected_layers"] = json.dumps(record["selected_layers"])
            writer.writerow(row)
        logging.info("Final result saved to %s", path)
    if cfg.results_json:
        path = Path(cfg.results_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = json.loads(path.read_text(encoding="utf-8")) if path.exists() and path.stat().st_size else []
        records.append(record)
        path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        logging.info("Final result saved to %s", path)


def main(cfg: AlignConfig):
    validate_result_files(cfg)
    configuration = asdict(cfg)
    if cfg.breakpoint_calibration:
        if not cfg.do_test or cfg.random_selection or cfg.do_layer_curve:
            raise ValueError("breakpoint_calibration requires do_test and cannot use random_selection or do_layer_curve")
        if not math.isfinite(cfg.top_layers) or not 0 < cfg.top_layers <= 1:
            raise ValueError("breakpoint_calibration requires 0 < top_layers <= 1")
        if cfg.calib_size < 1 or (cfg.limit is not None and cfg.limit < 0):
            raise ValueError("calib_size >= 1 and limit >= 0 are required")
        if not math.isfinite(cfg.mu) or not 0 <= cfg.mu <= 1 or not math.isfinite(cfg.sigma) or cfg.sigma <= 0:
            raise ValueError("mu must be between 0 and 1, and sigma must be positive and finite")
    set_seed(cfg.seed)
    os.makedirs(cfg.snapshot_path, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S_%f")
    run_name = generate_run_name(cfg) if cfg.run_name == "" else cfg.run_name
    run_name = f"{run_name}_{timestamp}"

    final_snapshot_path = os.path.join(cfg.snapshot_path, run_name)
    os.makedirs(final_snapshot_path, exist_ok=True)
    log_file_path = os.path.join(final_snapshot_path, "log.log")

    setup_logging(log_file_path=log_file_path, log_level=cfg.log_level)
    logging.info(f"Configuration: {cfg}")
    logging.info(f"All files (logs, models, metrics) will be saved to: {final_snapshot_path}")
    logging.info(f"Log level: {cfg.log_level}")
    log_gpu_info()

    # Initialize W&B
    if cfg.use_wandb:
        wandb_config = {
            k: v for k, v in cfg.__dict__.items() 
            if not k.startswith('wandb_')
        }
        
        wandb_tags = []
        if cfg.wandb_tags != "":
            wandb_tags = [tag.strip() for tag in cfg.wandb_tags.split(',')]
        
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            entity=cfg.wandb_entity,
            tags=wandb_tags,
            config=wandb_config
        )

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_B)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_A = AutoModelForCausalLM.from_pretrained(cfg.model_A, device_map={"": cfg.device}, torch_dtype=torch.float16, attn_implementation="sdpa")
    model_B = AutoModelForCausalLM.from_pretrained(cfg.model_B, device_map={"": cfg.device}, torch_dtype=torch.float16, attn_implementation="sdpa")
    model_A.eval()
    model_B.eval()

    # special case for Gemma
    if "gemma" in cfg.model_A.lower() or "gemma" in cfg.model_B.lower():
        torch._dynamo.config.cache_size_limit = 64

    model_A.name = cfg.model_A
    model_B.name = cfg.model_B

    evaluator = get_evaluator(cfg.test_task)
    
    if cfg.limit == 0:
        cfg.limit = None

    results = None
    if cfg.do_test_skyline:
        skyline_evaluator = SkylineEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length)
        results = skyline_evaluator.test(model_A, model_B, limit=cfg.limit)
    if cfg.do_test_baseline:
        baseline_evaluator = BaselineEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length)
        results = baseline_evaluator.test(model_A, model_B, limit=cfg.limit)
    if cfg.do_test:
        calibration_f1 = None
        communication_evaluator = CommunicationEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length)
        if cfg.top_layers > 0:
            text_config = getattr(model_A.config, "text_config", model_A.config)
            count = int(cfg.top_layers * text_config.num_hidden_layers)
            if not 1 <= count <= text_config.num_hidden_layers:
                raise ValueError("top_layers must select between one and all sender layers")
            if cfg.random_selection:
                if cfg.always_include_layer_0:
                    cfg.layers_list = [0, *random.sample(range(1, text_config.num_hidden_layers), count - 1)]
                else:
                    cfg.layers_list = random.sample(range(text_config.num_hidden_layers), count)
                logging.info(f"Randomly selected layers list: {cfg.layers_list}")
            else:
                with communication_model(model_A, model_B, cfg, layers_list=[-1], trace=True) as cv:
                    communication_evaluator.test(model_A, cv, limit=cfg.calib_size, no_wandb=True, do_calc_layer_importance=True)
                    # Receiver attention scores select sender cache layers.
                    importance = {sender: communication_evaluator.layer_importance_total[receiver]
                                  for sender, receiver in cv.layer_map.items()}
                if cfg.breakpoint_calibration:
                    calibration_f1 = calibrate_layer_sets(
                        importance, communication_evaluator, model_A, model_B, cfg, final_snapshot_path)
                else:
                    cfg = get_top_layers(importance, cfg)
        elif cfg.do_layer_curve:
            with communication_model(model_A, model_B, cfg, layers_list=cfg.layers_list, trace=True) as cv:
                communication_evaluator.test(model_A, cv, limit=cfg.calib_size, no_wandb=True, do_calc_layer_importance=True)
                importance = {sender: communication_evaluator.layer_importance_total[receiver]
                              for sender, receiver in cv.layer_map.items()}
            layer_ranking = get_layer_ranking(importance, cfg)
            if cfg.always_include_layer_0:
                layer_ranking = [0, *[int(layer) for layer in layer_ranking if layer != 0]]
        if not cfg.do_layer_curve:
            if cfg.breakpoint_calibration:
                set_seed(cfg.seed)
                logging.info("Final evaluation with calibrated layers %s; limit=%s", cfg.layers_list, cfg.limit)
            with communication_model(model_A, model_B, cfg, layers_list=cfg.layers_list) as cv:
                results = communication_evaluator.test(model_A, cv, limit=cfg.limit)
            save_communication_result(cfg, configuration, run_name, final_snapshot_path,
                                      cv.layers_list, results, calibration_f1)
        elif cfg.do_layer_curve:
            results = []
            for i in range(len(layer_ranking)):
                layers_list = layer_ranking[:i+1]
                logging.info(f"Evaluating with layers_list: {layers_list}")
                with communication_model(model_A, model_B, cfg, layers_list=layers_list) as cv:
                    result = communication_evaluator.test(model_A, cv, limit=cfg.limit)
                results.append(result)
            logging.info(f"Layer curve results: {results}")
            if cfg.use_wandb:
                wandb.log({f"layer_curve_results": results})
    if cfg.do_test_ac:
        ac = ActivationCommunicator(model_A, model_B, cfg.layer_k, cfg.layer_j, f=cfg.f).to(cfg.device)
        ac_evaluator = ACEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length)
        results = ac_evaluator.test(model_A, ac, limit=cfg.limit)
    if cfg.do_test_nld:
        nld_evaluator = NLDEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length, cfg.nld_max_tokens_model_A_and_B_phase1, cfg.sender_aware)
        results = nld_evaluator.test(model_A, model_B, limit=cfg.limit)
    if cfg.do_test_cipher:
        model_A = CipherAgent(model_A, tokenizer)
        model_B = CipherAgent(model_B, tokenizer)
        cipher_evaluator = CipherEvaluator(evaluator, tokenizer, cfg.use_wandb, cfg.max_input_length, cfg.nld_max_tokens_model_A_and_B_phase1, cfg.sender_aware)
        results = cipher_evaluator.test(model_A, model_B, limit=cfg.limit)
    # Finish W&B run
    if cfg.use_wandb:
        wandb.finish()
    return results


def parse_args(argv=None) -> AlignConfig:
    parser = argparse.ArgumentParser()
    for field, default in AlignConfig().__dict__.items():
        arg_type = type(default)
        if isinstance(default, bool):
            if default:
                parser.add_argument(f"--no_{field}", dest=field, action="store_false")
            else:
                parser.add_argument(f"--{field}", dest=field, action="store_true")
            parser.set_defaults(**{field: default})
        elif isinstance(default, list):
            element_type = type(default[0])
            parser.add_argument(f"--{field}", type=element_type, default=default, nargs="+")
        else:
            parser.add_argument(f"--{field}", type=arg_type, default=default)
    args = parser.parse_args(argv)
    return AlignConfig(**vars(args))



if __name__ == "__main__":
    config = parse_args()
    main(config)
