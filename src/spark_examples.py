import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.param import Param
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.tuning import TrainValidationSplit, TrainValidationSplitModel
from pyspark.sql import DataFrame
from xgboost.spark import SparkXGBRegressor
import sklearn.datasets

import flytekit
from flytekit import StructuredDataset, kwtypes, task, workflow
from flytekitplugins.deck.renderer import MarkdownRenderer
from flytekitplugins.papermill import NotebookTask
from flytekitplugins.spark import Spark

from .charts import plot_subgroup_performance
from .types import Dataset, ParamGrid, TrainXGBoostSparkOutput


@task(cache=True, cache_version="2")
def load_data() -> Dataset:
    features, target = sklearn.datasets.load_diabetes(
        return_X_y=True, as_frame=True, scaled=False
    )
    df = pd.concat([features, target], axis=1)

    return Dataset(
        data=StructuredDataset(dataframe=df),
        features=features.columns.tolist(),
        target=target.name,
    )


def extract_validation_metrics(
    tvs_model: TrainValidationSplitModel, paramGrid: List[Dict[Param, Any]]
) -> pd.DataFrame:
    # Extract evaluation metrics and corresponding parameters
    metrics = tvs_model.validationMetrics
    params = tvs_model.getEstimatorParamMaps()

    param_names = [p.name for p in paramGrid[0]]
    param_values = [[param_dict[p] for p in paramGrid[0]] for param_dict in params]

    results_df = pd.DataFrame(param_values, columns=param_names)
    results_df["metric"] = metrics

    return results_df


@task(
    task_config=Spark(
        spark_conf={
            "spark.driver.memory": "4g",
            "spark.executor.memory": "2g",
            "spark.executor.instances": "3",
            "spark.driver.cores": "3",
            "spark.executor.cores": "1",
        },
    ),
    disable_deck=False,
)
def train_xgboost_spark(
    dataset: Dataset, param_grid: ParamGrid
) -> TrainXGBoostSparkOutput:
    df = dataset.data.open(DataFrame).all()
    va = VectorAssembler(inputCols=dataset.features, outputCol="features")
    model = SparkXGBRegressor(features_col="features", label_col="target")

    param_grid = param_grid.build_params_grid(model)

    # combine the above two steps into a single pipeline
    pipeline = Pipeline(stages=[va, model])

    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=RegressionEvaluator(labelCol="target"),
        trainRatio=0.8,
    )

    # fit tvs and get the metrics
    tvs_model = tvs.fit(df)

    # Training Outputs
    best_pipeline = tvs_model.bestModel
    results_df = extract_validation_metrics(tvs_model, param_grid)
    scores_df = best_pipeline.transform(df).drop("features")
    feature_importance = dict(
        zip(
            dataset.features,
            best_pipeline.stages[-1].get_feature_importances().values(),
        )
    )

    return TrainXGBoostSparkOutput(
        best_pipeline=best_pipeline,
        results_df=results_df,
        scores_df=scores_df.toPandas(),
        feature_importance=feature_importance,
    )


@task(disable_deck=False)
def training_report(
    results_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    feature_importance: Dict[str, float],
):
    flytekit.Deck("Grid Search", results_df.to_html())
    fe_df = pd.DataFrame(feature_importance, index=[0]).T.sort_values(
        by=0, ascending=False
    )
    flytekit.Deck("Feature Importance", fe_df.to_html())
    flytekit.Deck("Subgroup Performance", plot_subgroup_performance(scores_df))

    mse = ((scores_df["target"] - scores_df["prediction"]) ** 2).mean()
    flytekit.Deck("MSE", MarkdownRenderer().to_html(f"# MSE\n{mse}"))


@task
def get_sd_uri(sd: StructuredDataset) -> str:
    return sd.literal.uri


notebook_training_report = NotebookTask(
    name="notebook_training_report",
    notebook_path=os.path.join(Path(__file__).parent.absolute(), "notebook.ipynb"),
    render_deck=True,
    disable_deck=False,
    inputs=kwtypes(results_path=str),
)


@workflow
def wf():
    param_grid = ParamGrid(
        max_depth=[3, 6], n_estimators=[10, 30, 100], learning_rate=[0.01, 0.2, 0.5]
    )

    dataset = load_data()
    train_outputs = train_xgboost_spark(dataset=dataset, param_grid=param_grid)
    training_report(
        results_df=train_outputs.results_df,
        scores_df=train_outputs.scores_df,
        feature_importance=train_outputs.feature_importance,
    )

    results_path = get_sd_uri(sd=train_outputs.results_df)
    notebook_training_report(results_path=results_path)
