"""PI0281 single reference-progress repair; no controller or actor changes."""
import torch
CALLS = 0


def update_s_batch(s_prev, ee_p, ee_R, traj_batch, window=0.15, M=16, lam=0.15):
    global CALLS
    CALLS += 1
    # The delivered candidates, window and metric are unchanged. Ordered local
    # projection preserves the first reachable basin of a self-overlapping path.
    from modules.trajectory import so3_log
    N = s_prev.shape[0]
    frac = torch.linspace(0.0, 1.0, M, device=s_prev.device)
    s_max = (s_prev + window).clamp(max=traj_batch.L)
    s_cand = s_prev.unsqueeze(-1) + frac.unsqueeze(0) * (s_max-s_prev).clamp_min(0.0).unsqueeze(-1)
    p_cand = traj_batch.p_at(s_cand)
    R_cand = traj_batch.R_at(s_cand)
    dp = p_cand-ee_p.unsqueeze(1)
    dR = torch.matmul(R_cand, ee_R.unsqueeze(1).transpose(-1, -2))
    dtheta = so3_log(dR.reshape(-1, 3, 3)).reshape(N, M, 3).norm(dim=-1)
    dist = torch.sqrt((dp*dp).sum(-1)+(lam*dtheta)**2)
    infinity = torch.full_like(dist[:, :1], float('inf'))
    left = torch.cat((infinity, dist[:, :-1]), dim=1)
    right = torch.cat((dist[:, 1:], infinity), dim=1)
    minima = (dist <= left) & (dist <= right)
    best = minima.to(torch.int64).argmax(dim=-1)
    s_new = torch.gather(s_cand, 1, best.unsqueeze(-1)).squeeze(-1)
    s_new = torch.maximum(s_new, s_prev)
    p_best = torch.gather(p_cand, 1, best.view(N, 1, 1).expand(-1, 1, 3)).squeeze(1)
    return s_new, (ee_p-p_best).norm(dim=-1)


def install():
    # The actual consumer imports inside _update_trajectory_progress on each
    # call. Bind the originating module in this process, not an unused alias.
    import modules.trajectory as consumer
    consumer.update_s_batch = update_s_batch
    assert consumer.update_s_batch is update_s_batch

def runtime_receipt():
    import modules.trajectory as consumer
    return dict(bound=consumer.update_s_batch is update_s_batch,calls=CALLS,
        consumer='modules.trajectory.update_s_batch',rule='FIRST_ORDERED_LOCAL_MINIMUM')
