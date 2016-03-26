import os


class Config:
    def __init__(self):

        # QUANDL SETUP

        self.stock_returns_db = "WIKI"
        self.stock_returns_col = '11'
        self.market_return_db = "YAHOO"
        self.market_returns_index = 'INDEX_GSPC'
        self.market_returns_column = '6'
        self.api_token = 'c54mBskiz_BsF4vWWL2s'
        self.returns_column_name = "Adj. Close"

        # PERIODS and REGRESSION SETUP

        self.start = '2010-10-01'
        self.actual_start = '2011-01-01'
        self.end = '2016-01-01'
        self.period = '2M'
        self.max_degree = 3
        self.df_reg_type = 'c'
        self.alpha = 0.05

        # DIRECTORIES

        # Root directory folder
        self.root = os.getcwd() + '/'
        self.data_folder = self.root + 'Data/'

        # folder to put periodic cluster data with computed csi
        # columns in those files are: SYMBOL, ClusterID, mean, PeriodID
        # csi is fixed and used as is
        self.clusters_folder_with_CSI = self.data_folder + 'cluster_data/'

        # folder to put daily observations with attached clusters:
        # columns in those files are: SYMBOL, TIMESTAMP_UTC, <sentiment_indicator_name>, ClusterID
        # CSI can be recomputed with any method, according to any custom period allocation
        self.clusters_folder_without_CSI = self.root + 'cluster_data_no_CSI/'
        # folder to put regression result files
        self.reg_results_folder = self.root + 'regression_results/'

        if not os.path.exists(self.reg_results_folder):
            os.makedirs(self.reg_results_folder)

        if not os.path.exists(self.clusters_folder_with_CSI):
            os.makedirs(self.clusters_folder_with_CSI)

        if not os.path.exists(self.clusters_folder_without_CSI):
            os.makedirs(self.clusters_folder_without_CSI)

        # Clusters with CSI file dictionary:
        self.csi_clusters_file_dict = {'Bearish Intensity': 'BearishIntensity.csv',
                                       'Bullish Intensity': 'BULLISH_INTENSITY.csv',
                                       'Bull Bear Message Ratio': 'BULL_BEAR_MSG_RATIO.csv',
                                       'Market Index': 'MarketIndex.csv'}  # TODO: Bull Minus Bear file missing

        # Clusters without computed CSI file dictionary:
        self.no_csi_clusters_file_dict = {'Bullish Intensity': 'Twitaggmat.csv'}  # TODO: add if need be


config = Config()
