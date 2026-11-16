"""
    cRealNVP with additional backward NLL and detailed-balance (DB) loss.
    Physics-agnostic: works with every DBLoss subclass (SurfaceDBLoss, CollisionDBLoss).

    Data format (key names are generic: v_i = condition, v_r = target of the kernel):
        data["main"] : v_i, v_r, v_i_bwd, v_r_bwd  -> NLL terms
        data["db"]   : see DBLoss                  -> DB term (only if db_loss is used)
"""
from ..db_mixin import DetailedBalanceMixin
from ..registry import register_model
from .realnvp import cRealNVP


@register_model("crealnvp_db")
class cRealNVPDB(DetailedBalanceMixin, cRealNVP):
    """cRealNVP trained with forward NLL, optional backward NLL and optional DB loss
    (see DetailedBalanceMixin)."""
