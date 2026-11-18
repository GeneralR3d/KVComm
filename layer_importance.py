from fractions import Fraction
from itertools import combinations

import torch
import numpy as np
import logging

def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float32)
    mn, mx = v.min(), v.max()
    if mx - mn < eps:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn + eps)

def gaussian_prior(L, mu, sigma=10.0):
    x = np.arange(L, dtype=np.float32)
    w = np.exp(-0.5 * ((x - mu) / float(sigma)) ** 2)
    return w

def calc_layer_importance(B_attn_weights: dict, model_A_name: str, layer_importance_total: dict):
    layer_importance = {}
    for i in B_attn_weights.keys():
        attn_weights = B_attn_weights[i]
        attn_matrix = attn_weights[..., 1:-attn_weights.shape[-2]]
        importance = attn_matrix.sum(dim=-1).mean()
        layer_importance[i] = 0 if torch.isnan(importance) else importance.item()
        layer_importance_total[i].append(layer_importance[i])
    return layer_importance_total

def get_top_layers(layer_importance_total: dict, cfg):
    topk_layers = get_layer_ranking(layer_importance_total, cfg)
    count = int(cfg.top_layers * len(topk_layers))
    if getattr(cfg, "always_include_layer_0", False):
        if count < 1:
            raise ValueError("top_layers would select zero layers")
        cfg.layers_list = [0, *[int(layer) for layer in topk_layers if layer != 0][:count - 1]]
    else:
        cfg.layers_list = topk_layers[:count]
    logging.info(f"Top {cfg.top_layers} from {len(topk_layers)} layers: {topk_layers}")
    logging.info(f"New layers list: {cfg.layers_list}")
    return cfg

def get_layer_scores(layer_importance_total: dict, cfg):
    importance = []
    n_layers = len(layer_importance_total.keys())
    assert n_layers == max(layer_importance_total.keys()) + 1
    for i in range(n_layers):
        importance.append(np.mean(layer_importance_total[i]))
    
    importance = normalize(importance)
    mu = cfg.mu * (n_layers - 1)
    gaussian = normalize(gaussian_prior(n_layers, mu=mu, sigma=cfg.sigma))
    return importance, gaussian


def get_layer_ranking(layer_importance_total: dict, cfg):
    importance, gaussian = get_layer_scores(layer_importance_total, cfg)
    importance = cfg.alpha * importance + (1.0 - cfg.alpha) * gaussian

    top_layers = np.argsort(importance)[::-1]
    logging.info(f"Layer ranking: {top_layers}")
    return top_layers


def enumerate_layer_candidates(attention, prior, count, *, always_include_layer_0=True):
    """Return an exact midpoint and IDs per distinct set, reserving layer 0 by default."""
    if len(attention) != len(prior) or not 1 <= count <= len(prior):
        raise ValueError("Matching layer scores and a count between 1 and the layer count are required")
    if always_include_layer_0 and count == 1:
        return [(Fraction(1, 2), (0,))]

    # Keep the breakpoints and rankings exact for the input scores, even when
    # a narrow interval has no representable floating-point midpoint.
    intercepts = [Fraction(float(score)) for score in prior]
    slopes = [Fraction(float(score)) - intercept for score, intercept in zip(attention, intercepts)]
    reserved = (0,) if always_include_layer_0 else ()
    layers = range(len(reserved), len(intercepts))
    breakpoints = {Fraction(0), Fraction(1)}
    for i, j in combinations(layers, 2):
        slope_difference = slopes[j] - slopes[i]
        if slope_difference:
            alpha = (intercepts[i] - intercepts[j]) / slope_difference
            if 0 < alpha < 1:
                breakpoints.add(alpha)

    candidates = {}
    points = sorted(breakpoints)
    for left, right in zip(points, points[1:]):
        alpha = (left + right) / 2
        # Python's stable sort gives a lower layer ID priority on a score tie.
        ranking = sorted(layers, key=lambda layer: intercepts[layer] + alpha * slopes[layer],
                         reverse=True)
        selected = (*reserved, *sorted(ranking[:count - len(reserved)]))
        candidates.setdefault(selected, alpha)
    return [(alpha, selected) for selected, alpha in candidates.items()]
