"""
StatX Scientific Platform Modules
Central registry for all statistical, AI, and scientific modules.
"""

import importlib

# ------------------------------------------------
# MODULE REGISTRY (Dynamic Loader)
# ------------------------------------------------

MODULES = {

    # CORE SYSTEMS
    "StatXStatisticsLibrary": "statx_global_statistics_library",
    "StatXWorkflowBuilder": "statx_workflow_builder",
    "StatXKnowledgeEngine": "statx_knowledge_engine",
    "StatXCloudPlatform": "statx_cloud_platform",
    "StatXGlobalDataNetwork": "statx_global_data_network",
    "StatXScientificSuperAI": "statx_scientific_super_ai",
    "StatXProfessionalInterface": "statx_professional_interface",

    # AI SYSTEMS
    "ai_statistical_advisor": "ai_statistical_advisor",
    "ai_discovery_lab": "ai_discovery_lab",
    "autonomous_scientific_discovery": "autonomous_scientific_discovery",

    # STATISTICS LABS
    "descriptive_lab": "descriptive_lab",
    "eda_lab": "eda_lab",
    "data_lab": "data_lab",
    "cleaning_lab": "cleaning_lab",
    "visualization_lab": "visualization_lab",

    # ANALYSIS
    "hypothesis_lab": "hypothesis_lab",
    "chi_square_lab": "chi_square_lab",
    "anova_lab": "anova_lab",
    "regression_lab": "regression_lab",

}


# ------------------------------------------------
# SAFE MODULE LOADER
# ------------------------------------------------

def load(module_name):

    if module_name not in MODULES:
        return None

    try:
        module_path = MODULES[module_name]
        module = importlib.import_module(f"modules.{module_path}")
        return module

    except Exception as e:
        print(f"Module load failed: {module_name} -> {e}")
        return None
