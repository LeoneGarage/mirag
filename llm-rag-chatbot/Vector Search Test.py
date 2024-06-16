# Databricks notebook source
dbutils.widgets.text("vs_endpoint_name", "", "Vector Search Endpoint Name")
dbutils.widgets.text("sitemap_urls", "", "URLs separated by comma of sitemap.xml")
dbutils.widgets.text(
    "accepted_domains", "", "Domain part of urls to process, separated by comma"
)
dbutils.widgets.text(
    "catalog", "", "Catalog name where data and vector search are stored"
)
dbutils.widgets.text(
    "schema_name", "", "Schema name where data and vector search are stored"
)

# COMMAND ----------

# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain llama-index databricks-vectorsearch pydantic==1.10.9 mlflow databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init-advanced $reset_all_data=false $vs_endpoint_name=$vs_endpoint_name $sitemap_urls=$sitemap_urls $accepted_domains=$accepted_domains $catalog=$catalog $schema_name=$schema_name

# COMMAND ----------

import math

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.documentation"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.documentation_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  index = vsc.create_direct_access_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embedding",
    schema={
     "id": "bigint",
     "url": "string",
     "content": "string",
     "embedding": "array<float>"}
  )
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  index = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# COMMAND ----------

serving_endpoint_name="bge-en-large-pt"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

import urllib
import json
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedEntityInput,
    ServedModelInputWorkloadSize,
    ServedModelInputWorkloadType,
    TrafficConfig,
    Route,
)
from datetime import timedelta

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
model_name_short = "bge_large_en_v1_5"
model_name = f"system.ai.{model_name_short}"
model_versions = client.search_model_versions(f"name='{model_name}'")
latest_model_version = max([int(v.version) for v in model_versions])

# TODO: use the sdk once model serving is available.
w = WorkspaceClient()
# Start the endpoint using the REST API (you can do it using the UI directly)
# auto_capture_config = {
#     "catalog_name": catalog,
#     "schema_name": db,
#     "table_name_prefix": serving_endpoint_name,
# }
# environment_vars = {"DATABRICKS_TOKEN": "{{secrets/dbdemos/rag_sp_token}}"}
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=model_name,
            entity_version=latest_model_version,
            name=f"{model_name_short}-{i}",
            workload_size="Large", #ServedModelInputWorkloadSize.LARGE,
            workload_type="GPU_SMALL", #ServedModelInputWorkloadType.GPU_SMALL,
            scale_to_zero_enabled=True,
            # environment_vars={
            #     "DATABRICKS_TOKEN": "{{secrets/dbdemos/rag_sp_token}}",  # <scope>/<secret> that contains an access token
            # },
        )
        for i in range(0, 5)
    ],
    traffic_config=TrafficConfig(
        routes=[
            Route(served_model_name=f"{model_name_short}-{i}",
                  traffic_percentage=100/5)
            for i in range(0, 5)
        ]
    )
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(
        f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint..."
    )
    w.serving_endpoints.create_and_wait(
        name=serving_endpoint_name,
        config=endpoint_config,
        route_optimized = True,
        timeout=timedelta(minutes=30),
    )
else:
    served_models = [
        m
        for m in existing_endpoint.config.served_entities
        if m.entity_name == model_name and int(m.entity_version) != latest_model_version
    ]
    print(served_models)
    if len(served_models) > 0:
        print(
            f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint..."
        )
        w.serving_endpoints.update_config_and_wait(
            served_entities=endpoint_config.served_entities,
            name=serving_endpoint_name,
            traffic_config=endpoint_config.traffic_config,
            timeout=timedelta(minutes=30),
        )

# COMMAND ----------

volume_folder =  f"/Volumes/{catalog}/{db}/volume_documentation"

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint=serving_endpoint_name, inputs={"input": batch})
        for e in response.data:
            yield e['embedding']

    def batch_embeddings(contents):
        # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
        max_batch_size = 150
        batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

        # Process each batch and collect the results
        for batch in batches:
            for e in get_embeddings(batch.tolist()):
                yield e

    return pd.Series(batch_embeddings(contents))

# COMMAND ----------

def upsert_index_udf(iterator):
    for pdf in iterator:
        ix = [o for o in pdf.to_dict('records')]
        for o in ix:
            o["embedding"] = [float(e) for e in o["embedding"]]
        index.upsert(ix)
        yield pdf

# COMMAND ----------

dbutils.fs.rm(f"dbfs:{volume_folder}/checkpoints/doc_index", True)

# COMMAND ----------

schema_df = spark.read.table("`safety-culture`.chatbot.documentation")
schema = schema_df.schema

# COMMAND ----------

doc_df = (spark.readStream.table("`safety-culture`.chatbot.documentation").repartition(int(math.ceil(schema_df.count() / 150)))
      .withColumn("embedding", get_embedding("content"))
      .mapInPandas(upsert_index_udf, schema=schema)
      .writeStream
      .format("noop")
      .trigger(availableNow=True)
      .option("checkpointLocation", f"dbfs:{volume_folder}/checkpoints/doc_index")
      .start()
      .awaitTermination())

# doc_df.write
#     .table('pdf_documentation').awaitTermination())
