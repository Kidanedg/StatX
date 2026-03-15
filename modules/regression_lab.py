import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


def run(df):

    st.title("📈 StatX Regression Analysis Laboratory")

    if df is None:
        st.warning("⚠ Upload dataset first.")
        return

    st.subheader("Dataset Overview")
    st.write(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    regression_type = st.selectbox(
        "Select Regression Type",
        [
            "Simple Linear Regression",
            "Multiple Linear Regression",
            "Logistic Regression",
            "Poisson Regression",
            "Stepwise Regression"
        ]
    )

    st.divider()

# --------------------------------------------------
# SIMPLE LINEAR REGRESSION
# --------------------------------------------------

    if regression_type == "Simple Linear Regression":

        x = st.selectbox("Independent Variable", numeric_cols)
        y = st.selectbox("Dependent Variable", numeric_cols)

        X = df[[x]]
        Y = df[y]

        model = LinearRegression()
        model.fit(X,Y)

        pred = model.predict(X)

        coef = model.coef_[0]
        intercept = model.intercept_
        mse = mean_squared_error(Y,pred)
        r2 = r2_score(Y,pred)

        st.subheader("Model Results")

        col1, col2 = st.columns(2)
        col1.metric("Coefficient", coef)
        col2.metric("Intercept", intercept)

        st.metric("Mean Squared Error", mse)
        st.metric("R²", r2)

        st.subheader("Regression Equation")

        st.latex(f"{y} = {intercept:.3f} + {coef:.3f}{x}")

        # Plot
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x], y=df[y], ax=ax)
        ax.plot(df[x], pred, color="red")
        st.pyplot(fig)

        # Interpretation
        st.subheader("Interpretation")

        direction = "increase" if coef > 0 else "decrease"

        st.write(f"""
        • The regression model explains **{r2*100:.2f}% of the variability in {y}**.

        • The coefficient of **{x} = {coef:.3f}** indicates that for every **one-unit increase in {x}**,  
        the predicted value of **{y} is expected to {direction} by {abs(coef):.3f} units**.

        • The intercept **{intercept:.3f}** represents the expected value of **{y} when {x}=0**.

        • The Mean Squared Error (MSE) of **{mse:.3f}** measures the average squared prediction error.
        """)

# --------------------------------------------------
# MULTIPLE LINEAR REGRESSION
# --------------------------------------------------

    elif regression_type == "Multiple Linear Regression":

        y = st.selectbox("Dependent Variable", numeric_cols)

        X_vars = st.multiselect("Independent Variables", numeric_cols)

        if len(X_vars) == 0:
            st.warning("Select predictors")
            return

        X = df[X_vars]
        Y = df[y]

        X = sm.add_constant(X)

        model = sm.OLS(Y,X).fit()

        st.subheader("Model Summary")
        st.text(model.summary())

        coef_table = pd.DataFrame({
            "Coefficient": model.params,
            "Std Error": model.bse,
            "t value": model.tvalues,
            "p value": model.pvalues
        })

        st.subheader("Coefficient Table")
        st.dataframe(coef_table)

        st.subheader("Model Fit")

        r2 = model.rsquared
        adjr2 = model.rsquared_adj

        st.write("R²:", r2)
        st.write("Adjusted R²:", adjr2)

        # Interpretation
        st.subheader("Interpretation")

        significant_vars = coef_table[coef_table["p value"] < 0.05].index.tolist()

        st.write(f"""
        • The regression model explains **{r2*100:.2f}% of the variation in {y}**.

        • After adjusting for number of predictors, the **Adjusted R² = {adjr2:.3f}**.

        • Variables with **p-value < 0.05** are statistically significant predictors.

        • Significant predictors detected: **{significant_vars}**

        • Positive coefficients indicate an increase in the dependent variable,
        while negative coefficients indicate a decrease.
        """)

# --------------------------------------------------
# LOGISTIC REGRESSION
# --------------------------------------------------

    elif regression_type == "Logistic Regression":

        y = st.selectbox("Binary Dependent Variable", df.columns)

        X_vars = st.multiselect("Predictor Variables", numeric_cols)

        if len(X_vars) == 0:
            st.warning("Select predictors")
            return

        X = df[X_vars]
        Y = df[y]

        model = LogisticRegression()
        model.fit(X,Y)

        pred = model.predict(X)

        accuracy = accuracy_score(Y,pred)

        coef_table = pd.DataFrame({
            "Variable":X_vars,
            "Coefficient":model.coef_[0]
        })

        st.subheader("Coefficient Table")
        st.dataframe(coef_table)

        st.metric("Model Accuracy", accuracy)

        # Interpretation
        st.subheader("Interpretation")

        st.write(f"""
        • Logistic regression models the **probability of occurrence of an event**.

        • Model classification accuracy is **{accuracy*100:.2f}%**.

        • A **positive coefficient** increases the probability of the outcome,
        while a **negative coefficient decreases it**.

        • Variables with larger absolute coefficients have stronger influence
        on the predicted probability.
        """)

# --------------------------------------------------
# POISSON REGRESSION
# --------------------------------------------------

    elif regression_type == "Poisson Regression":

        y = st.selectbox("Count Dependent Variable", numeric_cols)

        X_vars = st.multiselect("Predictors", numeric_cols)

        if len(X_vars) == 0:
            st.warning("Select predictors")
            return

        X = df[X_vars]
        Y = df[y]

        model = PoissonRegressor()
        model.fit(X,Y)

        coef_table = pd.DataFrame({
            "Variable":X_vars,
            "Coefficient":model.coef_
        })

        st.subheader("Poisson Coefficients")
        st.dataframe(coef_table)

        # Interpretation
        st.subheader("Interpretation")

        st.write("""
        • Poisson regression is used for **count data** such as number of events.

        • A positive coefficient increases the expected count of the outcome.

        • A negative coefficient decreases the expected count.

        • The exponential of coefficients represents the **multiplicative effect**
        on the expected count.
        """)

# --------------------------------------------------
# STEPWISE REGRESSION
# --------------------------------------------------

    elif regression_type == "Stepwise Regression":

        y = st.selectbox("Dependent Variable", numeric_cols)

        X_vars = st.multiselect("Candidate Predictors", numeric_cols)

        if len(X_vars) == 0:
            st.warning("Select predictors")
            return

        X = df[X_vars]
        Y = df[y]

        X = sm.add_constant(X)

        model = sm.OLS(Y,X).fit()

        pvals = model.pvalues

        selected = pvals[pvals < 0.05].index.tolist()

        st.subheader("Selected Variables")
        st.write(selected)

        st.text(model.summary())

        # Interpretation
        st.subheader("Interpretation")

        st.write(f"""
        • Stepwise regression selects predictors based on statistical significance.

        • Variables retained in the model (p < 0.05):

        {selected}

        • These variables contribute significantly to explaining variation in {y}.
        """)

# --------------------------------------------------
# RESIDUAL DIAGNOSTICS
# --------------------------------------------------

    st.divider()
    st.header("Residual Diagnostics")

    if 'pred' in locals():

        residuals = Y - pred

        fig, ax = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title("Residual Distribution")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=pred, y=residuals, ax=ax2)
        ax2.axhline(0,color="red")
        ax2.set_title("Residual vs Predicted")
        st.pyplot(fig2)

        st.write("""
        Interpretation:
        • Residuals should be approximately normally distributed.  
        • Random scatter around zero indicates a good model fit.  
        • Patterns suggest model misspecification.
        """)

    st.success("Regression analysis completed.")
