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

# MAGIC %md-sandbox
# MAGIC # Deploying our Chat Model and enabling Online Evaluation Monitoring
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-0.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC
# MAGIC
# MAGIC Let's now deploy our model as an endpoint to be able to send real-time queries.
# MAGIC
# MAGIC Once our model is live, we will need to monitor its behavior to detect potential anomaly and drift over time. 
# MAGIC
# MAGIC We won't be able to measure correctness as we don't have a ground truth, but we can track model perplexity and other metrics like profesionalism over time.
# MAGIC
# MAGIC This can easily be done by turning on your Model Endpoint Inference table, automatically saving every query input and output as one of your Delta Lake tables.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-leon_eller` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0417-051928-px0v0nze/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.27.0 mlflow==2.12.1 gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-init-advanced $reset_all_data=false $vs_endpoint_name=$vs_endpoint_name $sitemap_urls=$sitemap_urls $accepted_domains=$accepted_domains $catalog=$catalog $schema_name=$schema_name

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Deploy our model with Inference tables
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-1.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC Let's start by deploying our model endpoint.
# MAGIC
# MAGIC Simply define the `auto_capture_config` parameter during the deployment (or through the UI) to define the table where the endpoint request payload will automatically be saved.
# MAGIC
# MAGIC Databricks will fill the table for you in the background, as a fully managed service.

# COMMAND ----------

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

import urllib
import json
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
)
from datetime import timedelta

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
model_name = f"{catalog}.{db}.advanced-chatbot-model"
latest_model = client.get_model_version_by_alias(model_name, "prod")
latest_model_version = latest_model.version

# TODO: use the sdk once model serving is available.
w = WorkspaceClient()
serving_client = w.serving_endpoints
# serving_client = EndpointApiClient()
# Start the endpoint using the REST API (you can do it using the UI directly)
auto_capture_config = {
    "catalog_name": catalog,
    "schema_name": db,
    "table_name_prefix": serving_endpoint_name,
}
environment_vars = {"DATABRICKS_TOKEN": "{{secrets/dbdemos/rag_sp_token}}"}
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_TOKEN": "{{secrets/dbdemos/rag_sp_token}}",  # <scope>/<secret> that contains an access token
            },
        )
    ],
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
        timeout=timedelta(minutes=30),
    )
else:
    served_models = [
        m
        for m in existing_endpoint.config.served_models
        if m.model_name == model_name and m.model_version != latest_model_version
    ]
    if len(served_models) > 0:
        print(
            f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint..."
        )
        w.serving_endpoints.update_config_and_wait(
            served_models=endpoint_config.served_models,
            name=serving_endpoint_name,
            timeout=timedelta(minutes=30),
        )

# COMMAND ----------

displayHTML(
    f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.'
)

# COMMAND ----------

# DBTITLE 1,Let's try to send a query to our chatbot
response = serving_client.query(
    name=serving_endpoint_name,
    inputs=[
        {
            "messages": [
                {"role": "user", "content": "What is Company A"},
                {
                    "role": "assistant",
                    "content": "Company A is a company that provides applications for software developers.",
                },
                {"role": "user", "content": "Does it support creating projects?"},
            ]
        }
    ],
)
print(response.predictions[0])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Let's give it a try, using Gradio as UI!
# MAGIC
# MAGIC All you now have to do is deploy your chatbot UI. Here is a simple example using Gradio ([License](https://github.com/gradio-app/gradio/blob/main/LICENSE)). Explore the chatbot gradio [implementation](https://huggingface.co/spaces/databricks-demos/chatbot/blob/main/app.py).
# MAGIC
# MAGIC *Note: this UI is hosted and maintained by Databricks for demo purpose and is not intended for production use. We'll soon show you how to do that with Lakehouse Apps!*

# COMMAND ----------

# display_gradio_app("databricks-demos-chatbot")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Online LLM evaluation with Databricks Monitoring
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-online-2-2.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC Let's now analyze and monitor our model.
# MAGIC
# MAGIC
# MAGIC Here are the required steps:
# MAGIC
# MAGIC - Make sure the Inference table is enabled (it was automatically setup in the previous cell)
# MAGIC - Consume all the Inference table payload, and measure the model answer metrics (perplexity, complexity etc)
# MAGIC - Save the result in your metric table. This can first be used to plot the metrics over time
# MAGIC - Leverage Databricks Monitoring to analyze the metric evolution over time

# COMMAND ----------

# MAGIC %md
# MAGIC Online evaluation requires a couple steps to unpack the inference table output, compute the LLM metrics and turn on the Lakehouse Monitoring.
# MAGIC
# MAGIC Databricks provides a ready-to-use notebook that you can run directly to extract the data and setup the monitoring.
# MAGIC
# MAGIC Open the [05-Inference-Tables-Analysis-Notebook-with-LLM-Metrics]($./05-Inference-Tables-Analysis-Notebook-with-LLM-Metrics) notebook for more details, or just run it directly from this notebook:
# MAGIC
# MAGIC *Note that depending of your model input/output, you might need to change the notebook unpacking logic. See the notebook commments for more details*

# COMMAND ----------

# monitor = dbutils.notebook.run("./05-Inference-Tables-Analysis-Notebook-with-LLM-Metrics", 600,
#                             {"endpoint": serving_endpoint_name,
#                               "checkpoint_location": f'dbfs:/Volumes/{catalog}/{db}/volume_databricks_documentation/checkpoints/payload_metrics'})

# COMMAND ----------

# url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/sql/dashboards/{json.loads(monitor)["dashboard_id"]}'
# print(f"You can monitor the performance of your chatbot at {url}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ##Congratulations! You have learned how to automate GenAI application Analysis and Monitoring with Databricks!
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-eval-dashboard.png?raw=true" style="float:right" width="750px" />
# MAGIC
# MAGIC We have investigated the use of custom LLM metrics to track our Databricks Q&A chatbot model performance over time.
# MAGIC
# MAGIC Note that for a real use-case, you'll likely want to add a human feedback loop, reviewing where your model doesn't perform well (e.g. by providing your customer simple way to flag incorrect answers)
# MAGIC
# MAGIC This is also a good opportunity to either improve your documentation or adjust your prompt, and ultimately add the correct answer to your evaluation dataset!
# MAGIC
# MAGIC ### This concludes our Advanced chatbot demo
# MAGIC
# MAGIC In this demo, we covered:
# MAGIC
# MAGIC - How to ingest and extract information from unstructured documents
# MAGIC - Setup a self-managed vector search index in our Delta Table
# MAGIC - Build a more advanced LangChain model with history and filter
# MAGIC - Perform offline evaluation with LLM as a judge
# MAGIC - Deploy our new model endpoint with inference table
# MAGIC - Leverage Databricks Monitoring to track your model performance over time, and potentially trigger alarms when something is off.
# MAGIC
# MAGIC By bridging all these capabilities together, Databricks makes it easy to deploy your own RAG chatbot application with the Data Intelligence Platform!
