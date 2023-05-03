from flytekit import task, workflow, StructuredDataset, Resources
# from flytekit.image_spec import ImageSpec
from flytekitplugins.spark import Spark
import flytekit
from typing import Tuple
import pandas as pd

# python_image_spec = ImageSpec(packages=["numpy", "pandas"],
#                        apt_packages=["git"], 
#                        registry="peridotml")


# spark_image_spec = ImageSpec(base_image="ghcr.io/peridotml/mem_example:spark",
#                        packages=["numpy", "pandas", "pyspark", "flytekitplugins-spark", "flytekit"],
#                        apt_packages=["git"], 
#                        registry="peridotml")



import pandas as pd

from flytekit import task, workflow, StructuredDataset
from flytekitplugins.spark import Spark
import flytekit

def configure_s3(spark):
    hadoop_conf = spark._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoop_conf.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.WebIdentityTokenCredentialsProvider")


@task(container_image="ghcr.io/peridotml/mem_example:spark", task_config=Spark(
        spark_conf={
            "spark.driver.memory": "4g",
            "spark.executor.memory": "2g",
            "spark.executor.instances": "3",
            "spark.driver.cores": "3",
            "spark.executor.cores": "1",
        }),
    # limits=Resources(mem="30Gi", cpu="15"),
)
def t1() -> Tuple[StructuredDataset, str]:
    spark = flytekit.current_context().spark_session
    # configure_s3(spark)

    df = pd.DataFrame({"a": [1,2,3]})
    df = spark.createDataFrame(df)

    executor_count = len(spark.sparkContext._jsc.sc().statusTracker().getExecutorInfos()) - 1
    cores_per_executor = int(spark.sparkContext.getConf().get('spark.executor.cores','1'))

    return StructuredDataset(dataframe=df), str(spark.sparkContext.getConf().getAll())

@workflow
def wf() -> StructuredDataset:
    sd, conf = t1()
    return sd

# @task(container_image=python_image_spec)
# def t1() -> StructuredDataset:
#     df = pd.DataFrame({"a": [1,2,3]})
#     return StructuredDataset(dataframe=df)

# @task(container_image=python_image_spec)
# def t2(sd: StructuredDataset) -> int:
#     df = sd.open(pd.DataFrame).all()
#     return len(df)


# @workflow
# def wf() -> int:
#     sd, executor_count, cores_per_executor = t1()
#     return 10
#     # return t2(sd=sd)