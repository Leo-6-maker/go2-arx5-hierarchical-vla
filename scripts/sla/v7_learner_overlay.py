"""Build the isolated V7-A learner from the exact engine used in V6.5."""

import argparse
import hashlib
from pathlib import Path


ENGINE_SHA256 = "2e890ac06865ba30addfccbe02c9ecad5baeaffcb038d91ebb47ca658b995031"


def stage(source, destination):
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != ENGINE_SHA256:
        raise ValueError("The last V6.5 execution engine has changed")
    edits = (
        ("    support: np.ndarray | None = None\n",
         "    support: np.ndarray | None = None\n    condition: np.ndarray | None = None\n"),
        ('                 support_domains=("target",), optimizer_factory=None):\n',
         '                 support_domains=("target",), optimizer_factory=None,\n'
         '                 target_context=None, context_treatment=False):\n'),
        ("        self.parameters, self.frozen, self.rng = parameters, frozen, rng\n",
         "        self.target_context, self.context_treatment = target_context, context_treatment\n"
         "        self.parameters, self.frozen, self.rng = parameters, frozen, rng\n"),
        ("        make_optimizer = make_adamw if optimizer_factory is None else optimizer_factory\n",
         '        if target_context is not None:\n'
         '            from v7_full_model_paths import target_flow_loss, target_auxiliary\n'
         '            self.gradient["target"] = original.compile_fm_accumulation(target_flow_loss(graphs["target"]))\n'
         '            self.aux_add = original.compile_fm_accumulation(\n'
         '                target_auxiliary(graphs["target"], datasets["target"].statistics["actions"]))\n'
         "        make_optimizer = make_adamw if optimizer_factory is None else optimizer_factory\n"),
        ('            host = prepare_host_batch(reader, self.tokenizer, self.roles[domain],\n'
         '                                      range(cursor, cursor + len(indices)), indices)\n',
         '            cursors = range(cursor, cursor + len(indices))\n'
         '            condition = None\n'
         '            if domain == "target" and self.target_context is not None:\n'
         '                from v7_matched_bank import prepare_go2_batch\n'
         '                commands = {k: v[indices] for k, v in self.target_context.items()}\n'
         '                host, condition = prepare_go2_batch(reader, self.tokenizer, cursors, indices, commands,\n'
         '                                                    treatment=self.context_treatment)\n'
         '            else:\n'
         '                host = prepare_host_batch(reader, self.tokenizer, self.roles[domain], cursors, indices)\n'),
        ("        return PreparedBatch(host, indices, cursor, support)\n",
         "        return PreparedBatch(host, indices, cursor, support, condition)\n"),
        ("                noise, tau = self.draws[domain](self.rng, batch.cursor)\n",
         "                noise, tau = self.draws[domain](self.rng, batch.cursor)\n"
         "                condition = jax.device_put(batch.condition) if batch.condition is not None else None\n"),
        ("                        args = (*args, support[part])\n",
         "                        args = (*args, support[part])\n"
         "                    if condition is not None:\n"
         "                        args = (*args, condition[part])\n"),
        ("        selected = jax.tree.map(lambda x: x[jnp.asarray(slots)], observation)\n",
         "        selected = jax.tree.map(lambda x: x[jnp.asarray(slots)], observation)\n"
         "        condition = jax.device_put(batch.condition[slots]) if batch.condition is not None else None\n"),
        ("            loss, total = self.aux_add(total, micro / 4, active, self.frozen,\n"
         "                                       jax.tree.map(lambda x: x[part], selected),\n"
         "                                       noise[part], previous[part], label[part])\n",
         "            args = (jax.tree.map(lambda x: x[part], selected), noise[part], previous[part], label[part])\n"
         "            if condition is not None:\n"
         "                args = (*args, condition[part])\n"
         "            loss, total = self.aux_add(total, micro / 4, active, self.frozen, *args)\n"),
    )
    text = original.decode("utf-8")
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError("Expected a unique insertion point in the original training engine")
        text = text.replace(old, new)
    compile(text, str(destination), "exec")
    with destination.open("xb") as output:
        output.write(text.encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    stage(**vars(parser.parse_args()))
