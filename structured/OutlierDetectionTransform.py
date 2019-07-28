from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

class OutlierDetectionTransform:

    def __init__(self):
        self.__is_algorithm_set = False
        self.__algorithm = None
        self.__random_state = 1
        self.__jobs = -1
        self.__outliers_mask = []

    def set_random_state(self, state):
        self.__random_state = state

    def set_jobs(self, jobs):
        self.__jobs = jobs

    def set_isolation_forest(self, contamination_val):
        self.__is_algorithm_set = True
        self.__algorithm = IsolationForest(behaviour='new', n_jobs=self.__jobs, random_state=self.__random_state, contamination=contamination_val)

    def set_elliptic_envelope(self, contamination_val, support_fraction_val = None):
        self.__is_algorithm_set = True
        self.__algorithm = EllipticEnvelope(support_fraction=support_fraction_val, contamination=contamination_val, random_state=self.__random_state)

    def set_OCSVM(self, nu_val=0.5, gamma_val='scale', kernel_val='rbf', coef0_val=0.0):
        self.__is_algorithm_set = True
        self.__algorithm = OneClassSVM(nu=nu_val, kernel=kernel_val, coef0=coef0_val,
            gamma=gamma_val, shrinking=True)

    def filter_data(self, original_target, original_data):#, compare_data):

        if not self.__is_algorithm_set:
            return None, None

        self.__outliers_mask = self.__algorithm.fit_predict(original_data)
        data_filtered = original_data[self.__outliers_mask > 0]
        target_filtered = original_target[self.__outliers_mask > 0]
       # compare_filtered = compare_data[outliers_mask > 0]

        return target_filtered.copy(), data_filtered.copy()#, compare_filtered

    def mask_data(self, original_target, original_data):#, compare_data):

        if len(self.__outliers_mask) == 0:
            return None, None

        data_filtered = original_data[self.__outliers_mask > 0]
        target_filtered = original_target[self.__outliers_mask > 0]
       # compare_filtered = compare_data[outliers_mask > 0]

        return target_filtered.copy(), data_filtered.copy()#, compare_filtered