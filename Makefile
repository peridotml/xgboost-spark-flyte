# Set the base image name and repository
IMAGE_NAME := xgboost-examples
REPOSITORY := ghcr.io/peridotml

# Set the Docker tags for Spark and Ray
SPARK_TAG := $(REPOSITORY)/$(IMAGE_NAME):spark
RAY_TAG := $(REPOSITORY)/$(IMAGE_NAME):ray

# Define the build targets for Spark and Ray
.PHONY: build-spark build-ray build-all
build-spark:
	@docker build -t $(SPARK_TAG) -f Dockerfile.spark .

build-ray:
	@docker build -t $(RAY_TAG) -f Dockerfile.ray .

build-all: build-spark build-ray

# Define the push targets for Spark and Ray
.PHONY: push-spark push-ray push-all
push-spark: build-spark
	@docker push $(SPARK_TAG)

push-ray:	build-ray
	@docker push $(RAY_TAG)

push-all: push-spark push-ray