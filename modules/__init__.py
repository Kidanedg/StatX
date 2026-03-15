"""
StatX Scientific Platform Modules
Dynamic module registry
"""

import importlib

MODULES = {

# CORE SYSTEMS

"StatXStatisticsLibrary":"statx_global_statistics_library",
"StatXWorkflowBuilder":"statx_workflow_builder",
"StatXKnowledgeEngine":"statx_knowledge_engine",
"StatXCloudPlatform":"statx_cloud_platform",
"StatXGlobalDataNetwork":"statx_global_data_network",
"StatXScientificSuperAI":"statx_scientific_super_ai",
"StatXProfessionalInterface":"statx_professional_interface",

# AI SYSTEMS

"ai_statistical_advisor":"ai_statistical_advisor",
"ai_discovery_lab":"ai_discovery_lab",
"autonomous_scientific_discovery":"autonomous_scientific_discovery",

# STATISTICS LABS

"descriptive_lab":"descriptive_lab",
"eda_lab":"eda_lab",
"data_lab":"data_lab",
"cleaning_lab":"cleaning_lab",
"visualization_lab":"visualization_lab",
"advanced_graphs":"advanced_graphs",

# ANALYSIS

"hypothesis_lab":"hypothesis_lab",
"chi_square_lab":"chi_square_lab",
"anova_lab":"anova_lab",
"regression_lab":"regression_lab",

}


def load(module_name):

    try:

        module_path = MODULES[module_name]

        module = importlib.import_module(
            f"modules.{module_path}"
        )

        return module

    except Exception as e:

        return None"""
StatX Scientific Platform Modules
Central registry for all statistical, AI, and scientific modules.
"""

# ------------------------------------------------
# CORE STATX SYSTEMS
# ------------------------------------------------

from .statx_global_statistics_library import StatXStatisticsLibrary
from .statx_workflow_builder import StatXWorkflowBuilder
from .statx_knowledge_engine import StatXKnowledgeEngine
from .statx_cloud_platform import StatXCloudPlatform
from .statx_global_data_network import StatXGlobalDataNetwork
from .statx_scientific_super_ai import StatXScientificSuperAI
from .statx_professional_interface import StatXProfessionalInterface


# ------------------------------------------------
# AI SYSTEMS
# ------------------------------------------------

from .ai_statistical_advisor import ai_statistical_advisor
from .ai_discovery_lab import ai_discovery_lab
from .autonomous_scientific_discovery import autonomous_scientific_discovery


# ------------------------------------------------
# CORE STATISTICS LABS
# ------------------------------------------------

from .descriptive_lab import descriptive_lab
from .eda_lab import eda_lab
from .data_lab import data_lab
from .cleaning_lab import cleaning_lab
from .visualization_lab import visualization_lab
from .advanced_graphs import advanced_graphs


# ------------------------------------------------
# STATISTICAL ANALYSIS
# ------------------------------------------------

from .hypothesis_lab import hypothesis_lab
from .chi_square_lab import chi_square_lab
from .anova_lab import anova_lab
from .regression_lab import regression_lab
from .multicolinarty_test import multicollinearity_test
from .factor_lab import factor_lab
from .cluster_lab import cluster_lab
from .multivariate_lab import multivariate_lab


# ------------------------------------------------
# ADVANCED STATISTICS
# ------------------------------------------------

from .bayesian import bayesian
from .bayesian_lab import bayesian_lab
from .simulation_lab import simulation_lab
from .time_series_lab import time_series_lab
from .spatial_statistics_lab import spatial_statistics_lab
from .survival_lab import survival_lab


# ------------------------------------------------
# QUALITY & INDUSTRIAL STATISTICS
# ------------------------------------------------

from .qualitycontrol import qualitycontrol
from .quality_control_lab import quality_control_lab
from .experimental_design_lab import experimental_design_lab
from .reliablity_lifetime import reliability_lifetime


# ------------------------------------------------
# ECONOMETRICS & MACHINE LEARNING
# ------------------------------------------------

from .econometrics_lab import econometrics_lab
from .machine_learning_lab import machine_learning_lab


# ------------------------------------------------
# BIOSTATISTICS & BIOMEDICAL SCIENCE
# ------------------------------------------------

from .biostatistics import biostatistics
from .biostatistics_medical_lab import biostatistics_medical_lab
from .biometrics import biometrics
from .biometrics_modeling import biometrics_modeling


# ------------------------------------------------
# BIOINFORMATICS & OMICS
# ------------------------------------------------

from .bioinformatics import bioinformatics
from .genomics_dna_engine import genomics_dna_engine
from .systems_biology_omics_lab import systems_biology_omics_lab


# ------------------------------------------------
# MOLECULAR & CHEMICAL SCIENCE
# ------------------------------------------------

from .molecular_biology import molecular_biology
from .biomolecular_simulation import biomolecular_simulation
from .biomolecular_modeling_lab import biomolecular_modeling_lab
from .chemoinformatics import chemoinformatics
from .drug_discovery_lab import drug_discovery_lab


# ------------------------------------------------
# PHYSICS & ENERGY SYSTEMS
# ------------------------------------------------

from .statistical_physics import statistical_physics
from .bioenergy import bioenergy


# ------------------------------------------------
# SCIENTIFIC DISCOVERY SYSTEMS
# ------------------------------------------------

from .global_intelligence_lab import global_intelligence_lab


# ------------------------------------------------
# RESEARCH SYSTEMS
# ------------------------------------------------

from .report_generator import report_generator
from .regression_reporter import regression_reporter
from .research_paper_generator import research_paper_generator
from .research_reporting_lab import research_reporting_lab


# ------------------------------------------------
# CONSULTING SYSTEM
# ------------------------------------------------

from .stat_consultant import stat_consultant
