import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats


# -------------------------------------------------
# DATASET UNDERSTANDING ENGINE
# -------------------------------------------------

class DatasetInterpreter:

    def analyze_dataset(self, df):

        summary = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "numeric_variables": len(df.select_dtypes(include=np.number).columns),
            "categorical_variables": len(df.select_dtypes(exclude=np.number).columns)
        }

        return summary


# -------------------------------------------------
# KNOWLEDGE EXTRACTION ENGINE
# -------------------------------------------------

class KnowledgeExtractor:

    def extract_patterns(self, df):

        corr = df.corr(numeric_only=True)

        strong = []

        for i in corr.columns:
            for j in corr.columns:

                if i != j:

                    r = corr.loc[i, j]

                    if abs(r) > 0.6:
                        strong.append((i, j, r))

        return strong


# -------------------------------------------------
# LITERATURE KNOWLEDGE BASE
# -------------------------------------------------

class LiteratureKnowledgeBase:

    def __init__(self):

        self.knowledge = {

            "income-health": "Higher income improves health outcomes",
            "education-income": "Education increases earning potential",
            "exercise-health": "Regular exercise reduces disease risk",
            "pollution-health": "Air pollution increases respiratory diseases"

        }

    def search(self, keyword):

        results = []

        for k, v in self.knowledge.items():

            if keyword.lower() in k.lower():
                results.append(v)

        return results


# -------------------------------------------------
# RESEARCH COMPARISON ENGINE
# -------------------------------------------------

class ResearchComparator:

    def compare_with_literature(self, patterns, knowledge_base):

        comparisons = []

        for a, b, r in patterns:

            key = f"{a.lower()}-{b.lower()}"

            literature = knowledge_base.search(key)

            comparisons.append({

                "relationship": f"{a} vs {b}",
                "correlation": round(r, 3),
                "literature_support": literature

            })

        return comparisons


# -------------------------------------------------
# THEORY GENERATION ENGINE
# -------------------------------------------------

class TheoryGenerator:

    def generate_theories(self, patterns):

        theories = []

        for a, b, r in patterns:

            if r > 0:

                theories.append(
                    f"Theory: Increasing {a} may lead to increases in {b}."
                )

            else:

                theories.append(
                    f"Theory: Higher {a} may reduce {b}."
                )

        return theories


# -------------------------------------------------
# SCIENTIFIC INSIGHT ENGINE
# -------------------------------------------------

class ScientificInsight:

    def interpret_patterns(self, patterns):

        insights = []

        for a, b, r in patterns[:5]:

            insights.append(
                f"Strong statistical association detected between {a} and {b} (r={round(r,2)})."
            )

        return insights


# -------------------------------------------------
# MAIN STATX KNOWLEDGE ENGINE
# -------------------------------------------------

class StatXKnowledgeEngine:

    @staticmethod
    def run(df):

        st.title("StatX Global Scientific Knowledge Engine")

        if df is None:

            st.warning("Upload dataset first")
            return

        interpreter = DatasetInterpreter()
        extractor = KnowledgeExtractor()
        kb = LiteratureKnowledgeBase()
        comparator = ResearchComparator()
        theory = TheoryGenerator()
        insight = ScientificInsight()

        # ----------------------------------------------
        # DATASET INTERPRETATION
        # ----------------------------------------------

        st.header("Dataset Understanding")

        summary = interpreter.analyze_dataset(df)

        st.write(summary)

        # ----------------------------------------------
        # PATTERN EXTRACTION
        # ----------------------------------------------

        st.header("Scientific Pattern Discovery")

        patterns = extractor.extract_patterns(df)

        st.write(patterns)

        # ----------------------------------------------
        # LITERATURE COMPARISON
        # ----------------------------------------------

        st.header("Comparison With Scientific Literature")

        comparisons = comparator.compare_with_literature(patterns, kb)

        st.write(comparisons)

        # ----------------------------------------------
        # SCIENTIFIC INSIGHTS
        # ----------------------------------------------

        st.header("Scientific Insights")

        insights = insight.interpret_patterns(patterns)

        for i in insights:
            st.write("-", i)

        # ----------------------------------------------
        # THEORY GENERATION
        # ----------------------------------------------

        st.header("Generated Scientific Theories")

        theories = theory.generate_theories(patterns)

        for t in theories[:10]:
            st.write("-", t)
