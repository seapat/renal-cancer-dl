from typing import Callable
import pandas as pd
import numpy as np
from collections import UserDict

class CodeClassGeneSelector(UserDict):
    """Class for easy access to genes by their code class"""
    def __init__(self,df):
        if 'CodeClass' not in df.columns:
            raise ValueError(
                'df must be a gene DataFrame returned by '\
                'parse.get_rcc_data'
            )
        else:
            self.df = df
        self.gene_dict = (
            df.groupby('CodeClass')
            .apply(lambda x: x.to_dict('list'))
            .to_dict()
        )
        super().__init__(
            df.groupby('CodeClass')
            .apply(lambda x: x.drop('CodeClass',axis=1).to_dict('list'))
            .to_dict()
        )
    def get(self,code_class,field='Name'):
        return(self.gene_dict[code_class][field])

def _check_func(func):
    """Check the function to make sure it is valid"""
    if func == 'mean':
        func = np.mean
    elif func == 'median':
        func = np.median
    elif func == 'std':
        func = np.std
    elif not callable(func):
        raise TypeError(
            'func must be "mean" or "median" or a callable '\
                'function (e.g. scipy.stats.mstats.gmean)'
        )
    else:
        func = func
    return(func)
    
class FunctionGeneSelector():
    """Choose genes for normalization based on the data. For example you can 
    use the 100 genes with least standard deviation for normalization."""
    def __init__(self,genes,func:str|Callable='std',n=100,select_least=True):
        if 'CodeClass' not in genes.columns:
            raise ValueError(
                'genes must be a gene DataFrame returned by '\
                    'parse.get_rcc_data'
            )
        self.genes = genes
        self.func = _check_func(func)
        self.n = n
        self.select_least = select_least
    def get(self,df):
        if self.select_least:
            genes = df[self.genes['Name']].apply(self.func).nsmallest(self.n).index
        else:
            genes = df[self.genes['Name']].apply(self.func).nlargest(self.n).index
        return(genes)

class Normalize():
    def __init__(self,raw_data,genes):
        self.raw_data = raw_data
        self.genes = genes
        self.norm_data = raw_data.copy()
        self.norm_data.index = self.norm_data.SampleID.apply(lambda x: x.split(' ')[1])
        self.norm_data.index = self.norm_data.index.astype(int)
        self.norm_data = self.norm_data[genes['Name']]
    def _check_func(self,func):
        """Check the function to make sure it is valid"""
        if func == 'mean':
            func = np.mean
        elif func == 'median':
            func = np.median
        elif not callable(func):
            raise TypeError(
                'func must be "mean" or "median" or a callable '\
                    'function (e.g. scipy.stats.mstats.gmean)'
            )
        else:
            func = func
        return(func)
    def _check_genes(self,genes):
        if isinstance(genes,FunctionGeneSelector):
            genes = genes.get(self.norm_data)
        else:
            pass
        return(genes)
    def background_subtract(self,genes,func='mean',drop_genes=True):
        """Subtract background using negative controls."""
        func = self._check_func(func)
        genes = self._check_genes(genes)
        bg = self.norm_data[genes].apply(func,axis=1)
        if drop_genes:
            self.norm_data = self.norm_data.drop(genes,axis=1)
        self.norm_data = self.norm_data.subtract(bg,axis='index')
        return(self)
    def _scale_factor(self,genes,func='mean'):
        func = self._check_func(func)
        genes = self._check_genes(genes)
        scale_factor = func(self.norm_data[genes],axis=1)\
            /func(func(self.norm_data[genes],axis=1))
        return(scale_factor)
    def scale_by_genes(self,genes,func='mean',drop_genes=False):
        """Normalize against a set of genes usually positive controls or
        housekeeping genes."""
        sf = self._scale_factor(genes,func)
        genes = self._check_genes(genes)
        if drop_genes:
            self.norm_data = self.norm_data.drop(genes,axis=1)
        self.norm_data = self.norm_data.multiply(sf,axis='index')
        return(self)
    def quantile(self):
        """Performs Quantile normalization on a data frame where samples are rows
        and genes are columns."""
        m_rank = self.norm_data.rank(axis=1)
        m_sorted = self.norm_data.apply(
            lambda x: np.sort(x.values),axis=1,result_type='expand'
        )
        m_sorted.columns = self.norm_data.columns
        mean_vals = m_sorted.mean(axis=0)
        qnm = m_rank.apply(lambda x: np.interp(
            x,np.arange(1,m_rank.shape[1]+1),mean_vals),axis=1,
            result_type='expand'
        )
        qnm.columns = self.norm_data.columns
        self.norm_data = qnm
        return(self)
    def drop_genes(self,genes):
        """Remove genes from the normalized dataframe"""
        genes = self._check_genes(genes)
        self.norm_data.drop(genes,axis=1,inplace=True)
        return(self)
    def include_annot_cols(self):
        """Reattach the annotation columns from the raw data."""
        orig_cols = set(self.raw_data.columns)
        gene_cols = set(self.genes['Name'])
        annot_cols = orig_cols - gene_cols
        col_idx = self.raw_data.columns.isin(annot_cols)
        self.norm_data = pd.concat(
            [
                self.raw_data[self.raw_data.columns[col_idx]],
                self.norm_data
            ],
            axis=1
        )
        return(self)