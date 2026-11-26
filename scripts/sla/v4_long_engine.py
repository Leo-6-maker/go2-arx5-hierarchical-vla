"""One learner update for V4 mixed-source training or its matched control."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import time
from threading import Lock

import numpy as np


BATCH = {"target": 64, "robodojo": 32, "libero": 32}


@dataclass(frozen=True)
class PreparedBatch:
    host: dict
    indices: np.ndarray
    cursor: int
    support: np.ndarray | None = None


def masked_target_loss(graph):
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter.batch_v2 import error32

    def loss(active, frozen, observation, actions, noise, tau, support):
        actions = jnp.where(support[..., None], actions, 0.)
        with jax.default_matmul_precision("highest"):
            model = nnx.merge(graph, active["shared"], active["private"], frozen)
            error = error32(model, observation, actions, noise, tau)
        valid = jnp.where(support[..., None], error, 0.)
        return jnp.mean(jnp.sum(valid, axis=(-2, -1)) / (32 * jnp.sum(support, axis=-1)))

    return jax.jit(jax.value_and_grad(loss))


class LongLearner:
    """Keep one immutable visit plan, one model, and one global Adam clip."""

    def __init__(self, *, graphs, parameters, frozen, datasets, tokenizer, banks, auxiliary_slots,
                 auxiliary, cursors, rates, rng, target_control=False):
        from go2arx5_vla.sla_adapter import scale20k_training as original
        from go2arx5_vla.sla_adapter.adapt5k_training import OPTIMIZER
        from go2arx5_vla.sla_adapter.scale20k_contract import make_adamw

        self.parameters, self.frozen, self.rng = parameters, frozen, rng
        self.datasets, self.tokenizer, self.banks = datasets, tokenizer, banks
        self.auxiliary, self.auxiliary_slots = auxiliary, auxiliary_slots
        self.cursors, self.rates = cursors, rates
        self.domains = ("target",) if target_control else tuple(BATCH)
        self.gradient = {
            d: original.compile_fm_accumulation(
                masked_target_loss(graphs[d]) if d == "target" else original.domain_loss(graphs[d]))
            for d in self.domains
        }
        self.draws = {d: original.frozen_draws(d, BATCH[d]) for d in self.domains}
        self.aux_add = original.compile_fm_accumulation(
            original.target_auxiliary(graphs["target"], datasets["target"].statistics["actions"]))
        initialize, self.apply = make_adamw(**OPTIMIZER)
        self.optimizer = initialize(parameters)
        self.step = 0
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="v4-learner-input")
        self.reader_locks = {domain: Lock() for domain in self.domains}
        self.queued = {}

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def prepare(self, step, domain):
        from go2arx5_vla.sla_adapter.scale20k_training import prepare_host_batch

        indices = self.banks[domain][step]
        cursor = self.cursors[domain] + step * BATCH[domain]
        if len(indices) != BATCH[domain]:
            raise ValueError("Invalid realized domain batch")
        with self.reader_locks[domain]:
            reader = self.datasets[domain]
            host = prepare_host_batch(reader, self.tokenizer, domain,
                                      range(cursor, cursor + len(indices)), indices)
            if domain == "target":
                support = np.stack([reader.support(int(i)) for i in indices])
                if not support.any(axis=1).all():
                    raise ValueError("An optimized target sample has no actual consumption")
            else:
                support = None
        return PreparedBatch(host, indices, cursor, support)

    def prepared(self, step, stop):
        for k in range(step, min(step + 2, stop)):
            if k not in self.queued:
                self.queued[k] = {d: self.pool.submit(self.prepare, k, d) for d in self.domains}
        return self.queued.pop(step)

    def components(self, step, prepared, fm_micro, aux_micro, guard):
        import jax
        import jax.numpy as jnp
        from go2arx5_vla.sla_adapter.batch_v2 import device_batch
        from go2arx5_vla.sla_adapter.adapt5k_contract import joint_gradient

        if fm_micro not in (1, 4, 8, 16) or aux_micro not in (1, 2, 4):
            raise ValueError("Unsupported V4 microbatch")
        fm, metrics = {}, {}
        for domain in self.domains:
            guard()
            batch = prepared[domain].result()
            d0 = time.monotonic()
            obs, actions = device_batch(batch.host)
            support = jax.device_put(batch.support) if batch.support is not None else None
            noise, tau = self.draws[domain](self.rng, batch.cursor)
            active = {"shared": self.parameters["shared"], "private": self.parameters[domain]}
            total = jax.tree.map(jnp.zeros_like, active)
            losses = []
            for start in range(0, len(batch.indices), fm_micro):
                guard()
                part = slice(start, start + fm_micro)
                args = (jax.tree.map(lambda x: x[part], obs), actions[part], noise[part], tau[part])
                if support is not None:
                    args = (*args, support[part])
                loss, total = self.gradient[domain](total, fm_micro / len(batch.indices),
                                                    active, self.frozen, *args)
                losses.append(loss)
            fm[domain] = total
            metrics[domain] = dict(loss=float(jnp.mean(jnp.stack(losses))),
                                   visits=len(batch.indices),
                                   visit_sha256=hashlib.sha256(batch.indices.tobytes()).hexdigest(),
                                   seconds=time.monotonic() - d0)
            if domain == "target":
                aux, metrics["auxiliary"] = self.aux_gradient(step, batch, obs, active, aux_micro, guard)
        if self.domains == ("target",):
            joint = {key: jax.tree.map(lambda fm, a: 2. * fm + a, fm["target"][name], aux[name])
                     for key, name in (("shared", "shared"), ("target", "private"))}
        else:
            joint = joint_gradient(fm, aux, tree_map=jax.tree.map)
        return joint, {**fm, "auxiliary": aux}, metrics

    def aux_gradient(self, step, batch, observation, active, micro, guard):
        import jax
        import jax.numpy as jnp

        start = time.monotonic()
        slots = self.auxiliary_slots[step].tolist()
        if len(set(slots)) != 4 or min(slots) < 0 or max(slots) >= len(batch.indices):
            raise ValueError("Auxiliary conditions must be four distinct positions in the target batch")
        if not batch.support[slots].all():
            raise ValueError("Auxiliary generation requires complete real H40 support")
        previous, label = self.auxiliary(batch.indices[slots])
        if previous.shape != (4,) or label.shape != (4, 40):
            raise ValueError("Missing physical width and last consumed width")
        aux_noise = jax.random.fold_in(self.rng, 37)
        keys = jnp.stack([jax.random.fold_in(aux_noise, batch.cursor + slot) for slot in slots])
        noise = jax.vmap(lambda k: jax.random.normal(k, (40, 32)))(keys)
        previous, label = jax.device_put((previous, label))
        selected = jax.tree.map(lambda x: x[jnp.asarray(slots)], observation)
        total = jax.tree.map(jnp.zeros_like, active)
        losses = []
        for k in range(0, 4, micro):
            guard()
            part = slice(k, k + micro)
            loss, total = self.aux_add(total, micro / 4, active, self.frozen,
                                       jax.tree.map(lambda x: x[part], selected),
                                       noise[part], previous[part], label[part])
            losses.append(loss)
        return total, dict(loss=float(jnp.mean(jnp.stack(losses))), visits=4, slots=slots,
                           seconds=time.monotonic() - start)

    def update(self, stop, fm_micro, aux_micro, guard):
        import jax

        start = time.monotonic()
        joint, _, metrics = self.components(self.step, self.prepared(self.step, stop), fm_micro, aux_micro, guard)
        guard()
        rate = float(self.rates[self.step])
        parameters, optimizer, norm = self.apply(self.parameters, self.optimizer, joint, rate)
        jax.block_until_ready((parameters, optimizer))
        self.parameters, self.optimizer = parameters, optimizer
        self.step += 1
        return dict(step=self.step, lr=rate, global_gradient_norm=norm, components=metrics,
                    seconds=time.monotonic() - start)
