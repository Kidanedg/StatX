"""
StatX Core Scientific Infrastructure
Handles statistical engines, workflows, reporting, and the professional interface.
"""

# ------------------------------------------------
# STATISTICAL ENGINE
# ------------------------------------------------

from ..statx_global_statistics_library import StatXStatisticsLibrary


# ------------------------------------------------
# WORKFLOW SYSTEM
# ------------------------------------------------

from ..statx_workflow_builder import StatXWorkflowBuilder


# ------------------------------------------------
# REPORTING SYSTEM
# ------------------------------------------------

from ..report_generator import report_generator
from ..regression_reporter import regression_reporter


# ------------------------------------------------
# PROFESSIONAL USER INTERFACE
# ------------------------------------------------

from ..statx_professional_interface import StatXProfessionalInterface


# ------------------------------------------------
# EXPORTS
# ------------------------------------------------

__all__ = [

    "StatXStatisticsLibrary",
    "StatXWorkflowBuilder",
    "report_generator",
    "regression_reporter",
    "StatXProfessionalInterface"

]
