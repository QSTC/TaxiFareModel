# imports
from numpy.core.getlimits import MachArLike
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data
from TaxiFareModel.data import clean_data
from TaxiFareModel.ml_flow_base import MLFlowBase


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())])
        self.pipeline=pipe

    def run(self):
        '''returns a trained pipelined model'''
        """option 1, mais il faut que la méthode set_pipeline return pipe"""
        #self.pipeline=self.set_pipeline()
        #self.pipeline.fit(self.X, self.y)

        """option 2, quand la méthode set_pipeline met à jour l'attribut pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

    def train(self):

        # mlflow: create a run
        self.mlflow_create_run()

        # get the data
        X, y = get_data()

        # holdout
        X_train, X_test, y_train, y_test = self.holdout(X, y)

        # get the pipeline
        pipeline = self.set_pipeline()

        # fit the pipeline
        pipeline.fit(X_train, y_train)

        # evaluate the perf
        rmse = self.eval_perf(pipeline, X_test, y_test)

        # save the parameters and metrics of the model
        self.mlflow_log_param("data_size", 1000)
        self.mlflow_log_param("model_name", "linear regression")

        # save the rmse
        self.mlflow_log_metric("rmse", rmse)


if __name__ == "__main__":
    #df = get_data()
    #df=clean_data(df)
    #y = df["fare_amount"]
    #X = df.drop("fare_amount", axis=1)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #trainer = Trainer(X_train, y_train)
    #trainer.run()
    #trainer.evaluate(X_test, y_test)

    trainer=Trainer()
    trainer.train()
