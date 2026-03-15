import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class StatXStatisticsLibrary:

    def __init__(self, df):

        self.df = df
        self.num = df.select_dtypes(include=np.number)


# ------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------

    def mean(self):

        return self.num.mean()

    def median(self):

        return self.num.median()

    def mode(self):

        return self.num.mode()

    def variance(self):

        return self.num.var()

    def std(self):

        return self.num.std()

    def skewness(self):

        return stats.skew(self.num)

    def kurtosis(self):

        return stats.kurtosis(self.num)

    def coefficient_of_variation(self):

        return self.num.std()/self.num.mean()

    def quartiles(self):

        return self.num.quantile([0.25,0.5,0.75])

    def iqr(self):

        return stats.iqr(self.num)

# ------------------------------------------------
# PROBABILITY DISTRIBUTIONS
# ------------------------------------------------

    def normal_pdf(self,x,mu,sigma):

        return stats.norm.pdf(x,mu,sigma)

    def normal_cdf(self,x,mu,sigma):

        return stats.norm.cdf(x,mu,sigma)

    def binomial_pmf(self,k,n,p):

        return stats.binom.pmf(k,n,p)

    def poisson_pmf(self,k,lam):

        return stats.poisson.pmf(k,lam)

    def exponential_pdf(self,x,lam):

        return stats.expon.pdf(x,scale=1/lam)

    def gamma_pdf(self,x,a):

        return stats.gamma.pdf(x,a)

    def beta_pdf(self,x,a,b):

        return stats.beta.pdf(x,a,b)

# ------------------------------------------------
# HYPOTHESIS TESTS
# ------------------------------------------------

    def t_test_one_sample(self, sample, mu):

        return stats.ttest_1samp(sample,mu)

    def t_test_independent(self, x,y):

        return stats.ttest_ind(x,y)

    def paired_t_test(self,x,y):

        return stats.ttest_rel(x,y)

    def chi_square_test(self,observed):

        return stats.chisquare(observed)

    def ks_test(self,sample):

        return stats.kstest(sample,'norm')

    def mann_whitney(self,x,y):

        return stats.mannwhitneyu(x,y)

    def wilcoxon_test(self,x,y):

        return stats.wilcoxon(x,y)

# ------------------------------------------------
# ESTIMATION METHODS
# ------------------------------------------------

    def confidence_interval_mean(self,sample,alpha=0.05):

        mean = np.mean(sample)
        se = stats.sem(sample)
        ci = stats.t.interval(
            1-alpha,
            len(sample)-1,
            loc=mean,
            scale=se
        )

        return ci

# ------------------------------------------------
# REGRESSION MODELS
# ------------------------------------------------

    def linear_regression(self,X,y):

        model = LinearRegression()
        model.fit(X,y)

        return model

# ------------------------------------------------
# MULTIVARIATE STATISTICS
# ------------------------------------------------

    def pca(self,n_components=2):

        model = PCA(n_components=n_components)
        return model.fit_transform(self.num)

# ------------------------------------------------
# CLUSTERING
# ------------------------------------------------

    def kmeans(self,k=3):

        model = KMeans(n_clusters=k)

        return model.fit_predict(self.num)

# ------------------------------------------------
# CORRELATION METHODS
# ------------------------------------------------

    def pearson(self,x,y):

        return stats.pearsonr(x,y)

    def spearman(self,x,y):

        return stats.spearmanr(x,y)

    def kendall(self,x,y):

        return stats.kendalltau(x,y)

# ------------------------------------------------
# TIME SERIES METHODS
# ------------------------------------------------

    def moving_average(self,series,window):

        return series.rolling(window).mean()

    def exponential_smoothing(self,series,alpha=0.3):

        result = [series.iloc[0]]

        for i in range(1,len(series)):

            result.append(

                alpha*series.iloc[i]
                +(1-alpha)*result[i-1]

            )

        return result
