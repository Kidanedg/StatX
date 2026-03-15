import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


# -------------------------------------------------------
# STYLING ENGINE
# -------------------------------------------------------

def styled_table(df):

    return df.style \
        .background_gradient(cmap="Blues") \
        .set_properties(**{
            "font-size": "13px",
            "text-align": "center"
        })


# -------------------------------------------------------
# INTERPRETATION ENGINE
# -------------------------------------------------------

def interpret_anova(pvalue):

    if pvalue < 0.05:

        return """
**Conclusion**

The p-value is less than 0.05.

Therefore we reject the null hypothesis and conclude that
at least one group mean is statistically different.
"""

    else:

        return """
**Conclusion**

The p-value is greater than 0.05.

There is insufficient evidence to conclude that
the group means differ significantly.
"""


# -------------------------------------------------------
# ONE WAY ANOVA
# -------------------------------------------------------

def one_way_anova(df, response, factor):

    formula = f"{response} ~ C({factor})"

    model = smf.ols(formula, data=df).fit()

    table = anova_lm(model, typ=2)

    return model, table


# -------------------------------------------------------
# TWO WAY ANOVA
# -------------------------------------------------------

def two_way_anova(df, response, factor1, factor2):

    formula = f"{response} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"

    model = smf.ols(formula, data=df).fit()

    table = anova_lm(model, typ=2)

    return model, table


# -------------------------------------------------------
# FACTORIAL ANOVA
# -------------------------------------------------------

def factorial_anova(df, response, factors):

    formula = response + " ~ "

    for f in factors:
        formula += f"C({f}) + "

    for i in range(len(factors)):
        for j in range(i+1, len(factors)):
            formula += f"C({factors[i]}):C({factors[j]}) + "

    formula = formula[:-3]

    model = smf.ols(formula, data=df).fit()

    table = anova_lm(model, typ=2)

    return model, table


# -------------------------------------------------------
# MANOVA
# -------------------------------------------------------

def run_manova(df, responses, factor):

    formula = " + ".join(responses) + f" ~ C({factor})"

    maov = MANOVA.from_formula(formula, data=df)

    return maov.mv_test()


# -------------------------------------------------------
# REPEATED MEASURES ANOVA
# -------------------------------------------------------

def repeated_anova(df, subject, within, dv):

    model = sm.stats.AnovaRM(
        df,
        depvar=dv,
        subject=subject,
        within=[within]
    ).fit()

    return model


# -------------------------------------------------------
# TUKEY POSTHOC
# -------------------------------------------------------

def tukey_test(df, response, factor):

    tukey = pairwise_tukeyhsd(
        df[response],
        df[factor]
    )

    return pd.DataFrame(
        data=tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )


# -------------------------------------------------------
# BONFERRONI TEST
# -------------------------------------------------------

def bonferroni_test(pvalues):

    reject, pvals, _, _ = multipletests(
        pvalues,
        method="bonferroni"
    )

    return pvals


# -------------------------------------------------------
# INTERACTION PLOT
# -------------------------------------------------------

def interaction_plot(df, response, factor1, factor2):

    fig, ax = plt.subplots()

    sns.pointplot(
        data=df,
        x=factor1,
        y=response,
        hue=factor2,
        palette="viridis"
    )

    plt.title("Interaction Plot")

    st.pyplot(fig)


# -------------------------------------------------------
# DOE DESIGN GENERATOR
# -------------------------------------------------------

def factorial_design(levels):

    import itertools

    factors = list(levels.keys())

    design = list(
        itertools.product(*levels.values())
    )

    df = pd.DataFrame(design, columns=factors)

    return df


# -------------------------------------------------------
# MAIN STREAMLIT MODULE
# -------------------------------------------------------

def run(df):

    st.title("StatX Experimental Design & ANOVA Laboratory")

    if df is None:

        st.warning("Upload dataset first")

        return

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    categorical = df.select_dtypes(exclude=np.number).columns.tolist()

    analysis = st.selectbox(

        "Select Analysis",

        [
            "One Way ANOVA",
            "Two Way ANOVA",
            "Factorial ANOVA",
            "MANOVA",
            "Repeated Measures ANOVA",
            "Post Hoc Tests",
            "Interaction Plot",
            "Design of Experiments (DOE)"
        ]
    )

# -------------------------------------------------------
# ONE WAY ANOVA
# -------------------------------------------------------

    if analysis == "One Way ANOVA":

        response = st.selectbox("Response Variable", numeric)

        factor = st.selectbox("Factor", categorical)

        model, table = one_way_anova(df, response, factor)

        st.subheader("ANOVA Table")

        st.write(styled_table(table))

        pvalue = table["PR(>F)"][0]

        st.write(interpret_anova(pvalue))

# -------------------------------------------------------
# TWO WAY ANOVA
# -------------------------------------------------------

    elif analysis == "Two Way ANOVA":

        response = st.selectbox("Response", numeric)

        factor1 = st.selectbox("Factor 1", categorical)

        factor2 = st.selectbox("Factor 2", categorical)

        model, table = two_way_anova(df, response, factor1, factor2)

        st.write(styled_table(table))

# -------------------------------------------------------
# FACTORIAL ANOVA
# -------------------------------------------------------

    elif analysis == "Factorial ANOVA":

        response = st.selectbox("Response", numeric)

        factors = st.multiselect("Factors", categorical)

        model, table = factorial_anova(df, response, factors)

        st.write(styled_table(table))

# -------------------------------------------------------
# MANOVA
# -------------------------------------------------------

    elif analysis == "MANOVA":

        responses = st.multiselect("Response Variables", numeric)

        factor = st.selectbox("Factor", categorical)

        result = run_manova(df, responses, factor)

        st.text(result)

# -------------------------------------------------------
# REPEATED ANOVA
# -------------------------------------------------------

    elif analysis == "Repeated Measures ANOVA":

        subject = st.selectbox("Subject ID", df.columns)

        within = st.selectbox("Within Factor", categorical)

        dv = st.selectbox("Dependent Variable", numeric)

        model = repeated_anova(df, subject, within, dv)

        st.text(model)

# -------------------------------------------------------
# POST HOC
# -------------------------------------------------------

    elif analysis == "Post Hoc Tests":

        response = st.selectbox("Response", numeric)

        factor = st.selectbox("Factor", categorical)

        tukey = tukey_test(df, response, factor)

        st.subheader("Tukey HSD Results")

        st.write(styled_table(tukey))

# -------------------------------------------------------
# INTERACTION PLOT
# -------------------------------------------------------

    elif analysis == "Interaction Plot":

        response = st.selectbox("Response", numeric)

        factor1 = st.selectbox("Factor 1", categorical)

        factor2 = st.selectbox("Factor 2", categorical)

        interaction_plot(df, response, factor1, factor2)

# -------------------------------------------------------
# DOE GENERATOR
# -------------------------------------------------------

    elif analysis == "Design of Experiments (DOE)":

        st.subheader("Factorial Design Generator")

        k = st.number_input("Number of Factors", 2, 5)

        levels = {}

        for i in range(k):

            name = st.text_input(f"Factor {i+1} Name")

            level = st.number_input(
                f"Levels for {name}",
                2,
                5
            )

            levels[name] = list(range(1, level+1))

        design = factorial_design(levels)

        st.write("Experimental Design Matrix")

        st.dataframe(design)

    st.success("Experimental analysis completed")
