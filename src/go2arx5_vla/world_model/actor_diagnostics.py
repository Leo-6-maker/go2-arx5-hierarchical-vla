"""Distillation diagnostics; model-teacher scores are not physical rewards."""
import numpy as np


def distributions(logits, labels, scores, *, high_margin_m):
    logits, labels, scores = (np.asarray(v, dtype=np.float64) for v in (logits, labels, scores))
    if (logits.shape != labels.shape or scores.shape != labels.shape or logits.ndim != 2
            or logits.shape[1] != 4 or not len(logits) or not all(np.isfinite(v).all() for v in (logits, labels, scores))
            or np.any(labels < 0) or not np.allclose(labels.sum(1), 1.) or high_margin_m <= 0):
        raise ValueError('FINITE_FOUR_ACTION_TEACHER_DISTRIBUTION_REQUIRED')
    logp = logits-logits.max(1, keepdims=True)
    logp -= np.log(np.exp(logp).sum(1, keepdims=True))
    logq = np.log(np.maximum(labels, np.finfo(float).tiny))
    entropy = -(labels*logq).sum(1)
    kl = (labels*(logq-logp)).sum(1)
    chosen, target = logits.argmax(1), labels.argmax(1)
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1]-sorted_scores[:, -2]
    high = margin >= high_margin_m
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (target, chosen), 1)
    regret = scores.max(1)-scores[np.arange(len(chosen)), chosen]
    return dict(examples=len(logits), confusion_teacher_rows_student_columns=confusion.tolist(),
                teacher_argmax_agreement=float(np.mean(chosen == target)),
                teacher_entropy=float(entropy.mean()), teacher_student_KL=float(kl.mean()),
                soft_cross_entropy=float((entropy+kl).mean()),
                mean_model_regret_m=float(regret.mean()),
                margin_weighted_model_regret_m=float(np.mean(regret*margin/high_margin_m)),
                high_margin_threshold_m=high_margin_m, high_margin_examples=int(high.sum()),
                high_margin_error_rate=float(np.mean(chosen[high] != target[high])) if high.any() else None,
                selected_action_counts=np.bincount(chosen, minlength=4).tolist(),
                teacher_action_counts=np.bincount(target, minlength=4).tolist(),
                physical_control_benefit='NOT_ESTABLISHED_BY_DISTILLATION')


def grouped_diagnostics(logits, labels, scores, parents, regions, *, high_margin_m):
    parents, regions = np.asarray(parents), np.asarray(regions)
    if parents.shape != (len(logits),) or regions.shape != parents.shape:
        raise ValueError('EXPLICIT_EXAMPLE_PARENT_AND_GOAL_REGION_REQUIRED')
    result = distributions(logits, labels, scores, high_margin_m=high_margin_m)
    for name, values in [('by_parent', parents), ('by_goal_region', regions)]:
        result[name] = {str(v): distributions(logits[values == v], labels[values == v], scores[values == v],
                         high_margin_m=high_margin_m) for v in np.unique(values)}
    return result


def normalizer_diagnostics(inputs, mean, scale):
    inputs, mean, scale = (np.asarray(v, dtype=np.float64) for v in (inputs, mean, scale))
    if inputs.ndim != 2 or mean.shape != inputs.shape[1:] or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError('FROZEN_TRAIN_NORMALIZER_LAYOUT_REQUIRED')
    z = np.abs((inputs-mean)/scale)
    if not np.isfinite(z).all():
        raise ValueError('NONFINITE_NORMALIZED_INPUT')
    return dict(absolute_z_quantiles={str(q): float(np.quantile(z, q)) for q in (.5, .95, .99, 1.)},
                fraction_absolute_z_above_5=float(np.mean(z > 5)),
                fraction_absolute_z_above_10=float(np.mean(z > 10)))
