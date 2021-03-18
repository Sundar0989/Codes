import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats.stats as stats
import pandas.core.algorithms as algos
#from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

pd.options.mode.chained_assignment = None 

class PSI(BaseEstimator, TransformerMixin):
    
    """Population Stability Index for categorical variables. For numeric variables, 
    it automatically bins them into number of bins as specified by user. 
    
    Parameters
    ----------
    date: column_name 
        Provide the column name which has the date field. 
    
    input_col: column_name 
        Provide the column name which has the input field with categories. If the input column is numeric, 
        then bins option will be used to create equal buckets and used for further analysis. 
        Also, set the var_type as 'num' when the input column is numeric.
        
    count: column_name 
        Provide the column name which has the count for each category.  
        
    bins: int (default=5)
        Bin numerical column into category based on this number specified. It creates
        bins with equal distribution of data in each bin for the first (start) date. The same
        bins is used for rest of PSI calculation.
    
    var_type: 'char' or 'num' (default='char')
        Specify the variable type for category option in this column.
        'char' - character variable
        'num' - numeric variable
    """
    
    # Initialize the parameters for the function
    def __init__(self, date, input_col, count, bins=5, var_type='char'):
        
        self.date = date
        self.input_col = input_col
        self.count = count
        assert isinstance(bins, int), "Expects int input. " + str(type(bins)) + " provided."
        assert bins > 1, "Bin value should be greater than 1. Provided value is " + str(bins) + "." 
        self.bins = bins
        assert var_type in ['char', 'num'], "Var type should be 'char' or 'num'. Provided option - " + str(var_type) + "."
        self.var_type = var_type
    
    # psi index calculation method
    @staticmethod
    def psi_index(actual, expected):
        actual = actual + 0.0001 
        expected = expected + 0.0001
        diff = actual - expected
        ln_value = np.log(actual/expected)
        index = diff*ln_value
        return index
    
    # check input data type - Only Pandas Dataframe allowed
    def check_datatype(self, X):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("The input data must be pandas dataframe. But the input provided is " + str(type(X)))
        return self
        
    # the fit function for PSI transformer
    def fit(self, X, y=None):
        
        #check datatype of X
        self.check_datatype(X)
        
        # check data type for input_col when numeric
        if self.var_type == 'num':
            assert np.issubdtype(X[self.input_col], np.number), "Input column is not numeric. Please check your input data type."
            sub_X = X[X[self.date] == X[self.date].min()] # select the subset data based on the start date in dataframe
            _, self.bins_to_apply = pd.qcut(sub_X[self.input_col], self.bins, retbins=True) # apply quantile binning on numerical variable
            self.num_bins = len(self.bins_to_apply) - 1 # unique bins in numerical variable
        else:
            self.bins_to_apply = X[self.input_col].unique() 
            self.num_bins = X[self.input_col].nunique() # unique bins in categorical variable
            
        # check number of bins > 1
        assert self.num_bins > 1, "Less number of categories to perform PSI. Number of bins - " + str(num_bins) + ". Please check your input."
                
        return self
    
    # Calculate metrics - This function is applied to dataframe to calculate monitoring metrics.

    def _calculate_metrics(self, X, date, category, value):
        
        # calculate count for each of the category and also include the groups that are missing
        c = X.pivot_table(values=value, index=category, columns=date, aggfunc=np.sum, fill_value=0).unstack()
        c = c.astype('float')
        # calculate percentage distribution by date
        pcts = c/c.groupby(level=0).sum()
        
        # number of unique values in date and category
        mul_value = X[date].nunique()
        shift_value = X[category].nunique()
        check_category_list = X[category].unique()
        
        # check the category are same as present in fit function
        assert set(check_category_list).issubset(self.bins_to_apply), "New category found in the data. Please check."

        # creating previous day counts and percentage
        shift_counts = c.shift(shift_value)
        shift_pcts = pcts.shift(shift_value)

        # concatenate data
        agg = pd.concat([c, pcts, shift_counts, shift_pcts], axis=1)
        agg.columns = ['count', 'percent', 'prev_count', 'prev_percent']
        agg = agg.reset_index()

        # create count and percent values for each group
        agg['dev_count'] = list(agg['count'][:shift_value])*mul_value
        agg['dev_percent'] = list(agg['percent'][:shift_value])*mul_value
        agg['prev_count'] = agg['prev_count'].fillna(agg['dev_count'])
        agg['prev_percent'] = agg['prev_percent'].fillna(agg['dev_percent'])
        
        # to make sure we dont divide by zero
        agg = agg.replace({'count': 0, 'dev_count': 0, 'prev_count': 0}, 1)

        #create delta values for each category
        agg['dev_delta'] = ((agg['count'] - agg['dev_count'])/agg['dev_count']) * 100
        agg['prev_delta'] = ((agg['count'] - agg['prev_count'])/agg['prev_count']) * 100

        # create psi values for each category
        agg['dev_psi_index'] = self.psi_index(agg['percent'], agg['dev_percent'])
        agg['prev_psi_index'] = self.psi_index(agg['percent'], agg['prev_percent'])
        agg['dev_psi'] = agg.groupby(date)['dev_psi_index'].transform('sum')
        agg['prev_psi'] = agg.groupby(date)['prev_psi_index'].transform('sum')
        
        # aggregated PSI values
        psi_agg = agg.groupby(date).agg({'dev_psi': 'mean', 'prev_psi': 'mean'}).reset_index()

        return agg, psi_agg
        
    #Transform new data or existing data based on the bins fit identified.
    def transform(self, X, y=None):
        
        self.check_datatype(X) #check input datatype. 
        
        # check for the fit variable
        #check_is_fitted(self, 'num_bins')
        if not self.num_bins:
            raise ValueError("Estimator has to be fitted to calculate PSI metrics.")
            
        # convert numerical variable to categories based on bins identified before
        if self.var_type == 'num':
            X[self.input_col] = pd.cut(X[self.input_col], self.bins_to_apply)

        #calculate PSI metrics and relative deltas
        outX = self._calculate_metrics(X, self.date, self.input_col, self.count)
            
        #transformed dataframe 
        return outX
    
    #Method that describes what we need this transformer to do
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
