# Databricks notebook source
import requests

# COMMAND ----------

company_name = "Gilbert + Tobin" #Company name for which rag data and model is being generated
accepted_domains = "www.gtlaw.com.au" # Comma separated list of accepted domains to scrape, all other will be ignored, e.g. www.databricks.com for Databricks
catalog = "gilbert_tobin" # Catalog name where rag data and model will live
schema_name = "chatbot" # Schema name in catalog where rag data and model will live
sitemap_urls = "https://www.gtlaw.com.au/sitemap.xml" # Comma separated list
vs_endpoint_name = "vs_endpoint" # e.g. vs_index
model_prompt = f"You are a trustful assistant for {company_name} customers, as well as {company_name} company. You are answering questions about {company_name}, {company_name} concepts, {company_name} people, {company_name} company, {company_name} Applications, or other {company_name} topics and other information related to {company_name}. If {company_name} word appears by itself, assume the question is about the company. If {company_name} word is not in the question, assume the question is about {company_name} company and {company_name} related topics"

# COMMAND ----------

logo_url = "https://www.gtlaw.com.au/sites/default/files/GT_landscape_logo_positive_RBG_0.png"
example_q1 = f"What is {company_name}?"
example_q2 = "List questions I can ask?"
example_q3 = "What information can you give me about {company_name} work with Guzman y Gomez?"
example_q4 = "Please provide a paragraph about {company_name} work with Blackstone on investment"

# COMMAND ----------

notebook_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
email = spark.sql('select current_user() as user').collect()[0]['user']
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

rag_job_payload = {
    "name": f"{company_name} website RAG",
    "email_notifications": {"no_alert_for_skipped_runs": False},
    "webhook_notifications": {},
    "timeout_seconds": 0,
    "max_concurrent_runs": 1,
    "tasks": [
        {
            "task_key": "ingest_pdf_documents",
            "run_if": "ALL_SUCCESS",
            "notebook_task": {
                "notebook_path": f"/Workspace/Users/{email}/mirag/llm-rag-chatbot/02-advanced/01-PDF-Advanced-Data-Preparation",
                "source": "WORKSPACE",
            },
            "job_cluster_key": "Job_cluster",
            "timeout_seconds": 0,
            "email_notifications": {},
            "notification_settings": {
                "no_alert_for_skipped_runs": False,
                "no_alert_for_canceled_runs": False,
                "alert_on_last_attempt": False,
            },
            "webhook_notifications": {},
        },
        {
            "task_key": "scrape_web_pages",
            "run_if": "ALL_SUCCESS",
            "notebook_task": {
                "notebook_path": f"/Workspace/Users/{email}/mirag/llm-rag-chatbot/01-quickstart/01-Data-Preparation-and-Index",
                "base_parameters": {"url_prefix": ""},
                "source": "WORKSPACE",
            },
            "job_cluster_key": "Job_cluster",
            "timeout_seconds": 0,
            "email_notifications": {},
            "notification_settings": {
                "no_alert_for_skipped_runs": False,
                "no_alert_for_canceled_runs": False,
                "alert_on_last_attempt": False,
            },
            "webhook_notifications": {},
        },
        {
            "task_key": "create_vector_index",
            "depends_on": [
                {"task_key": "scrape_web_pages"},
                {"task_key": "ingest_pdf_documents"},
            ],
            "run_if": "ALL_SUCCESS",
            "notebook_task": {
                "notebook_path": f"/Workspace/Users/{email}/mirag/llm-rag-chatbot/02-advanced/Prepare-Vector-Search-Index",
                "source": "WORKSPACE",
            },
            "job_cluster_key": "Job_cluster",
            "timeout_seconds": 0,
            "email_notifications": {},
            "notification_settings": {
                "no_alert_for_skipped_runs": False,
                "no_alert_for_canceled_runs": False,
                "alert_on_last_attempt": False,
            },
            "webhook_notifications": {},
        },
        {
            "task_key": "create_model_chain",
            "depends_on": [{"task_key": "create_vector_index"}],
            "run_if": "ALL_SUCCESS",
            "notebook_task": {
                "notebook_path": f"/Workspace/Users/{email}/mirag/llm-rag-chatbot/02-advanced/02-Advanced-Chatbot-Chain",
                "base_parameters": {"model_prompt": model_prompt},
                "source": "WORKSPACE",
            },
            "job_cluster_key": "Job_cluster",
            "timeout_seconds": 0,
            "email_notifications": {},
            "notification_settings": {
                "no_alert_for_skipped_runs": False,
                "no_alert_for_canceled_runs": False,
                "alert_on_last_attempt": False,
            },
            "webhook_notifications": {},
        },
        {
            "task_key": "deploy_model_endpoint",
            "depends_on": [{"task_key": "create_model_chain"}],
            "run_if": "ALL_SUCCESS",
            "notebook_task": {
                "notebook_path": f"/Workspace/Users/{email}/mirag/llm-rag-chatbot/02-advanced/04-Deploy-Model-as-Endpoint",
                "source": "WORKSPACE",
            },
            "job_cluster_key": "Job_cluster",
            "timeout_seconds": 0,
            "email_notifications": {},
            "notification_settings": {
                "no_alert_for_skipped_runs": False,
                "no_alert_for_canceled_runs": False,
                "alert_on_last_attempt": False,
            },
            "webhook_notifications": {},
        },
    ],
    "job_clusters": [
        {
            "job_cluster_key": "Job_cluster",
            "new_cluster": {
                "cluster_name": "",
                "spark_version": "14.3.x-scala2.12",
                "aws_attributes": {
                    "first_on_demand": 1,
                    "availability": "SPOT_WITH_FALLBACK",
                    "zone_id": "auto",
                    "spot_bid_price_percent": 100,
                    "ebs_volume_count": 0,
                },
                "node_type_id": "i3.xlarge",
                "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"},
                "enable_elastic_disk": True,
                "data_security_mode": "SINGLE_USER",
                "runtime_engine": "STANDARD",
                "autoscale": {"min_workers": 2, "max_workers": 64},
            },
        }
    ],
    "parameters": [
        {"name": "accepted_domains", "default": accepted_domains},
        {"name": "catalog", "default": catalog},
        {"name": "schema_name", "default": schema_name},
        {"name": "sitemap_urls", "default": sitemap_urls},
        {"name": "vs_endpoint_name", "default": vs_endpoint_name},
    ],
    "run_as": {"user_name": email},
}

# COMMAND ----------

bot_job_payload = {
    "name": f"{company_name} Chatbot",
    "email_notifications": {"no_alert_for_skipped_runs": False},
    "webhook_notifications": {},
    "timeout_seconds": 0,
    "max_concurrent_runs": 1,
    "tasks": [
        {
            "task_key": "run-gradio-chat",
            "run_if": "ALL_SUCCESS",
            "notebook_task": {
                "notebook_path": f"/Workspace/Users/{email}/mirag/llm-rag-chatbot/02-advanced/advanced-gradio-ui",
                "base_parameters": {
                    "example_q3": example_q3,
                    "catalog": catalog,
                    "example_q2": example_q2,
                    "logo_url": logo_url,
                    "schema_name": schema_name,
                    "example_q1": example_q1,
                    "example_q4": example_q4,
                },
                "source": "WORKSPACE",
            },
            "job_cluster_key": "Job_cluster",
            "timeout_seconds": 0,
            "email_notifications": {},
            "notification_settings": {
                "no_alert_for_skipped_runs": False,
                "no_alert_for_canceled_runs": False,
                "alert_on_last_attempt": False,
            },
            "webhook_notifications": {},
        }
    ],
    "job_clusters": [
        {
            "job_cluster_key": "Job_cluster",
            "new_cluster": {
                "cluster_name": "",
                "spark_version": "14.3.x-scala2.12",
                "spark_conf": {
                    "spark.master": "local[*, 4]",
                    "spark.databricks.cluster.profile": "singleNode",
                },
                "aws_attributes": {
                    "first_on_demand": 1,
                    "availability": "SPOT_WITH_FALLBACK",
                    "zone_id": "auto",
                    "spot_bid_price_percent": 100,
                    "ebs_volume_count": 0,
                },
                "node_type_id": "m5d.xlarge",
                "driver_node_type_id": "m5d.xlarge",
                "custom_tags": {"ResourceClass": "SingleNode"},
                "spark_env_vars": {"PYSPARK_PYTHON": "/databricks/python3/bin/python3"},
                "enable_elastic_disk": True,
                "data_security_mode": "SINGLE_USER",
                "runtime_engine": "STANDARD",
                "num_workers": 0,
            },
        }
    ],
    "run_as": {"user_name": email},
}

# COMMAND ----------

def send_job_request(action, request_func):
    api_url = host
    token = notebook_token

    response = request_func(f'{api_url}/api/2.1/jobs/{action}', {"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        raise Exception("Error: %s: %s" % (response.json()["error_code"], response.json()["message"]))
    return response.json()

# COMMAND ----------

send_job_request('create', lambda u, h: requests.post(f'{u}', json=rag_job_payload, headers=h))

# COMMAND ----------

send_job_request('create', lambda u, h: requests.post(f'{u}', json=bot_job_payload, headers=h))
