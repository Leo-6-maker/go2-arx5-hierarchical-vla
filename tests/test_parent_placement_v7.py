from go2arx5_vla.sla_adapter.model_v1 import placement_dtype_name


def test_restore_precision_retains_interfaces_and_frozen_dense_leaves():
    assert placement_dtype_name(('params','action_in_proj','kernel','value'))=='float32'
    assert placement_dtype_name(('params','action_out_proj','bias','value'))=='float32'
    assert placement_dtype_name(('params','time_mlp_in','kernel','value'))=='float32'
    assert placement_dtype_name(('params','PaliGemma','llm','layers','lora_b','value'))=='float32'
    assert placement_dtype_name(('params','PaliGemma','img','embedding','kernel','value'))=='bfloat16'
    assert placement_dtype_name(('params','PaliGemma','llm','layers','kernel','value'))=='bfloat16'
