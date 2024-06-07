# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1/ Data preparation for LLM Chatbot RAG
# MAGIC
# MAGIC ## Building and indexing our knowledge base into Databricks Vector Search
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-managed-flow-1.png?raw=true" style="float: right; width: 800px; margin-left: 10px">
# MAGIC
# MAGIC In this notebook, we'll ingest our documentation pages and index them with a Vector Search index to help our chatbot provide better answers.
# MAGIC
# MAGIC Preparing high quality data is key for your chatbot performance. We recommend taking time to implement these next steps with your own dataset.
# MAGIC
# MAGIC Thankfully, Lakehouse AI provides state of the art solutions to accelerate your AI and LLM projects, and also simplifies data ingestion and preparation at scale.
# MAGIC
# MAGIC For this example, we will use Databricks documentation from [docs.databricks.com](docs.databricks.com):
# MAGIC - Download the web pages
# MAGIC - Split the pages in small chunks of text
# MAGIC - Compute the embeddings using a Databricks Foundation model as part of our Delta Table
# MAGIC - Create a Vector Search index based on our Delta Table  
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F01-quickstart%2F01-Data-Preparation-and-Index&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-quickstart%2F01-Data-Preparation-and-Index&version=1">

# COMMAND ----------

# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-leon_eller` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0417-051928-px0v0nze/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# DBTITLE 1,Install required external libraries 
# MAGIC %pip install mlflow==2.10.1 lxml transformers==4.30.2 langchain==0.1.5 databricks-vectorsearch==0.22 beautifulsoup4
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Init our resources and catalog
# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Extracting Databricks documentation sitemap and pages
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-data-prep-1.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC First, let's create our raw dataset as a Delta Lake table.
# MAGIC
# MAGIC For this demo, we will directly download a few documentation pages from `docs.databricks.com` and save the HTML content.
# MAGIC
# MAGIC Here are the main steps:
# MAGIC
# MAGIC - Run a quick script to extract the page URLs from the `sitemap.xml` file
# MAGIC - Download the web pages
# MAGIC - Use BeautifulSoup to extract the ArticleBody
# MAGIC - Save the HTML results in a Delta Lake table

# COMMAND ----------

if not table_exists("raw_documentation_v4") or spark.table("raw_documentation_v4").isEmpty():
    # Download Databricks documentation to a DataFrame (see _resources/00-init for more details)
    doc_articles = download_documentation_articles()
    #Save them as a raw_documentation table
    doc_articles.write.mode('overwrite').saveAsTable("raw_documentation_v4")
    # doc_articles.write.mode('overwrite').saveAsTable("`safety-culture`.debug.links")

# display(spark.table("raw_documentation_v4").limit(2))

# COMMAND ----------

urls = [r.url for r in spark.read.format("delta").table("`safety-culture`.debug.links2").collect()]

# COMMAND ----------

def fetch_html(http, url):
    try:
      print(f"Fetching {url}")
      with http.get(url, headers={"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"}) as response:
        print(f"Fetched {url}, status_code {response.status_code}")
        if response.status_code == 200:
          ct = response.headers.get("Content-Type", None)
          if "text/html" in ct:
            return response.content
    except requests.RequestException as e:
      print(f"Exception fetching {url}, {str(e)}")
      return None
    return None


# COMMAND ----------

adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
try:
    with requests.Session() as http:
        http.mount("http://", adapter)
        http.mount("https://", adapter)
        for u in urls:
          html = fetch_html(http, u)
finally:
    adapter.close()


# COMMAND ----------

adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
try:
    with requests.Session() as http:
        http.mount("http://", adapter)
        http.mount("https://", adapter)
        # for u in urls:
        html = fetch_html(http, "http://safetyculture.com")
finally:
    adapter.close()

