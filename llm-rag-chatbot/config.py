# Databricks notebook source
dbutils.widgets.text("vs_endpoint_name", "", "Vector Search Endpoint Name")
dbutils.widgets.text("sitemap_urls", "", "URLs separated by comma of sitemap.xml")
dbutils.widgets.text("accepted_domains", "", "Domain part of urls to process, separated by comma")
dbutils.widgets.text("catalog", "", "Catalog name where data and vector search are stored")
dbutils.widgets.text("schema_name", "", "Schema name where data and vector search are stored")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Configuration file
# MAGIC
# MAGIC Please change your catalog and schema here to run the demo on a different catalog.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2Fconfig&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2Fconfig&version=1">

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME=dbutils.widgets.get("vs_endpoint_name")#"one-env-shared-endpoint-3"

SITEMAP_URLS = [u.strip() for u in dbutils.widgets.get("sitemap_urls").split(",")]#["https://safetyculture.com/sitemap.xml"]
accepted_domains = [d.strip() for d in dbutils.widgets.get("accepted_domains").split(",")]#["safetyculture.com", "safetyculture.io"]

catalog = dbutils.widgets.get("catalog")#"safety-culture"

#email = spark.sql('select current_user() as user').collect()[0]['user']
#username = email.split('@')[0].replace('.', '_')
#dbName = db = f"rag_chatbot_{username}"
dbName = db = dbutils.widgets.get("schema_name")#"chatbot"

# COMMAND ----------

# MAGIC %md
# MAGIC ### License
# MAGIC This demo installs the following external libraries on top of DBR(ML):
# MAGIC
# MAGIC
# MAGIC | Library | License |
# MAGIC |---------|---------|
# MAGIC | langchain     | [MIT](https://github.com/langchain-ai/langchain/blob/master/LICENSE)     |
# MAGIC | lxml      | [BSD-3](https://pypi.org/project/lxml/)     |
# MAGIC | transformers      | [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)     |
# MAGIC | unstructured      | [Apache 2.0](https://github.com/Unstructured-IO/unstructured/blob/main/LICENSE.md)     |
# MAGIC | llama-index      | [MIT](https://github.com/run-llama/llama_index/blob/main/LICENSE)     |
# MAGIC | tesseract      | [Apache 2.0](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE)     |
# MAGIC | poppler-utils      | [MIT](https://github.com/skmetaly/poppler-utils/blob/master/LICENSE)     |
# MAGIC | textstat      | [MIT](https://pypi.org/project/textstat/)     |
# MAGIC | tiktoken      | [MIT](https://github.com/openai/tiktoken/blob/main/LICENSE)     |
# MAGIC | evaluate      | [Apache2](https://pypi.org/project/evaluate/)     |
# MAGIC | torch      | [BDS-3](https://github.com/intel/torch/blob/master/LICENSE.md)     |
# MAGIC | tiktoken      | [MIT](https://github.com/openai/tiktoken/blob/main/LICENSE)     |
# MAGIC
# MAGIC
# MAGIC
# MAGIC