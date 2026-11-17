"""V5.2 clean-proxy physical endpoint objective, never an action processor."""

def endpoint_loss(clean, tau, q01, q99, *, xp):
    xyz=(clean[...,:3]+1)*.5*(q99[:3]-q01[:3]+1e-6)+q01[:3]
    differences=xp.stack((xyz[:,1]-xyz[:,0],xyz[:,39]-xyz[:,38]),axis=1)
    square=xp.sum(differences*differences,axis=-1)
    # Exact norm away from zero; the inactive square-root branch stays finite
    # so reverse mode has an explicitly zero derivative at a zero displacement.
    norm=xp.where(square>0,xp.sqrt(xp.where(square>0,square,xp.ones_like(square))),xp.zeros_like(square))
    acceleration=norm/(.05**2)
    terms=xp.mean(xp.maximum((acceleration-.20)/.25,0)**2,axis=-1)
    return xp.mean(terms*(tau<=.5).astype(terms.dtype))


def clean_estimate(x_tau, tau, field):
    return x_tau-tau[:,None,None]*field
