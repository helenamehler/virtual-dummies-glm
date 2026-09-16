# UPSTREAM, EXTENDED: taulantkoka/virtual-dummies (Koka et al., 2026), modified for GLM/Cox score-based selection (Mehler, Koka, Muma, 2026). GPLv3, see NOTICE.
from .vd_selectors import (
    VD_LARS, VD_OMP, VD_AFS, VD_AFS_Logistic, VD_AFS_Poisson, VD_AFS_Multinomial,
    VD_AFS_Cox,
    VDOptions, VDDummyLaw, ActiveFeature, MMapMatrix,
    TRexSelector, TRexOptions, TRexResult,
    SolverType, CalibMode,
)

__all__ = [
    "VD_LARS", "VD_OMP", "VD_AFS", "VD_AFS_Logistic", "VD_AFS_Poisson",
    "VD_AFS_Multinomial", "VD_AFS_Cox",
    "VDOptions", "VDDummyLaw", "ActiveFeature", "MMapMatrix",
    "TRexSelector", "TRexOptions", "TRexResult",
    "SolverType", "CalibMode",
]
