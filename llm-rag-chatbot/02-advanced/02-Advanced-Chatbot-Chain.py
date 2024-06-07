# Databricks notebook source
dbutils.widgets.text("vs_endpoint_name", "", "Vector Search Endpoint Name")
dbutils.widgets.text("sitemap_urls", "", "URLs separated by comma of sitemap.xml")
dbutils.widgets.text("accepted_domains", "", "Domain part of urls to process, separated by comma")
dbutils.widgets.text("catalog", "", "Catalog name where data and vector search are stored")
dbutils.widgets.text("schema_name", "", "Schema name where data and vector search are stored")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # 2/ Advanced chatbot with message history and filter using Langchain and DBRX Instruct
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-flow-2.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">
# MAGIC
# MAGIC Our Vector Search Index is now ready!
# MAGIC
# MAGIC Let's now create a more advanced langchain model to perform RAG.
# MAGIC
# MAGIC We will improve our langchain model with the following:
# MAGIC
# MAGIC - Build a complete chain supporting a chat history, using Databricks DBRX Instruct input style
# MAGIC - Add a filter to only answer Databricks-related questions
# MAGIC - Compute the embeddings with Databricks BGE models within our chain to query the self-managed Vector Search Index
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=%2F02-advanced%2F02-Advanced-Chatbot-Chain&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F02-advanced%2F02-Advanced-Chatbot-Chain&version=1">
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-leon_eller` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0417-051928-px0v0nze/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# MAGIC %pip install mlflow==2.12.1 lxml==4.9.3 langchain==0.1.17 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.27.0 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC %pip install pip mlflow[databricks]==2.12.1 gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-init-advanced $reset_all_data=false $vs_endpoint_name=$vs_endpoint_name $sitemap_urls=$sitemap_urls $accepted_domains=$accepted_domains $catalog=$catalog $schema_name=$schema_name

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploring Langchain capabilities
# MAGIC
# MAGIC Let's start with the basics and send a query to a Databricks Foundation Model using LangChain.

# COMMAND ----------

# DBTITLE 1,Spark Chat Model Prompt
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a detailed answer to this question: {question}"
)
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500, temperature = 0.8, extra_params = {
    "top_p": 0.95
  })

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
print(chain.invoke({"question": "What is SafetyCulture?"}))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Adding conversation history to the prompt 

# COMMAND ----------

prompt_with_history_str = """
You are a chatbot for SafetyCulture users and customers and SafetyCulture company. You are answering all questions related to SafetyCulture, SafetyCulture people, SafetyCulture Inspections, SafetyCulture company, SafetyCulture Application, or other SafetyCulture topics. If SafetyCulture word appears by itself, assume the question is about the company. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as detailed as possible.

Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""

prompt_with_history = PromptTemplate(
  input_variables = ["chat_history", "question"],
  template = prompt_with_history_str
)

# COMMAND ----------

# MAGIC %md When invoking our chain, we'll pass history as a list, specifying whether each message was sent by a user or the assistant. For example:
# MAGIC
# MAGIC ```
# MAGIC [
# MAGIC   {"role": "user", "content": "What is Apache Spark?"}, 
# MAGIC   {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
# MAGIC   {"role": "user", "content": "Does it support streaming?"}
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC Let's create chain components to transform this input into the inputs passed to `prompt_with_history`.

# COMMAND ----------

# DBTITLE 1,Chat History Extractor Chain
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | chat_model
    | StrOutputParser()
)

print(chain_with_history.invoke({
    "messages": [
        {"role": "user", "content": "What is SafetyCulture?"}, 
        {"role": "assistant", "content": "SafetyCulture is a company that provides tooling and applications for workplace safety and site inspections"}, 
        {"role": "user", "content": "Does it support creating custom templates?"}
    ]
}))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Let's add a filter on top to only answer Databricks-related questions.
# MAGIC
# MAGIC We want our chatbot to be profesionnal and only answer questions related to Databricks. Let's create a small chain and add a first classification step. 
# MAGIC
# MAGIC *Note: this is a fairly naive implementation, another solution could be adding a small classification model based on the question embedding, providing faster classification*

# COMMAND ----------

# DBTITLE 1,Databricks Inquiry Classifier
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500, temperature = 0.8, extra_params = {
    "top_p": 0.95
  })

is_question_about_safetyculture_str = """
You are classifying documents to know if this question is related with SafetyCulture Application users and customers, as well as SafetyCulture company, SafetyCulture concepts, SafetyCulture employees, SafetyCulture people, SafetyCulture Inspections, SafetyCulture company, SafetyCulture Application, other SafetyCulture topics and other information related with SafetyCulture, or something from a very different field. Also answer no if the last part is inappropriate. If SafetyCulture word appears by itself, assume the question is about the company. Do not add explanation to the answer. Also answer questions if they are in relation to altering responses with different style or humor or rhyming.

Here are some examples:

Question: Knowing this followup history: What is SafetyCulture?, classify this question: Do you have more details?
Expected Response: Yes

Knowing this followup history: What is SafetyCulture?, classify this question: what is the revenue
Expected Response: Yes

Question: Knowing this followup history: What is SafetyCulture?, classify this question: what jobs are advertised
Expected Response: Yes

Question: Knowing this followup history: What is SafetyCulture?, classify this question: Say it with humour
Expected Response: Yes

Question: Knowing this followup history: What is SafetyCulture?, classify this question: Say it as a limerick
Expected Response: Yes

Question: Knowing this followup history: What is SafetyCulture?, classify this question: What is NUIX?.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_safetyculture_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_safetyculture_str
)

is_about_safetyculture_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_safetyculture_prompt
    | chat_model
    | StrOutputParser()
)

#Returns "Yes" as this is about Databricks: 
print(is_about_safetyculture_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is SafetyCulture?"}, 
        {"role": "assistant", "content": "SafetyCulture is a company that provides tooling and applications for workplace safety and sire inspections"}, 
        {"role": "user", "content": "Does it support creating custom templates?"}
    ]
}))

# COMMAND ----------

# print(is_about_safetyculture_chain.invoke({
#     "messages": [
#         {"role": "user", "content": "What is SafetyCulture?"}, 
#         {"role": "assistant", "content": "SafetyCulture is a company that provides tooling and applications for workplace safety and sire inspections"}, 
#         {"role": "user", "content": "who is the coo"}
#     ]
# }))

# COMMAND ----------

#Return "no" as this isn't about Databricks
print(is_about_safetyculture_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}))

# COMMAND ----------

chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500, temperature = 0.8, extra_params = {
    "top_p": 0.95
  })

is_question_about_model_str = """
You are classifying commands to determine if the user wants to use a different LLM model. Respond with Databricks model endpoint name when the user requests a particular model.

Here are some examples:

Command: Use DBRX
Expected Response: databricks-dbrx-instruct

Command: Use Databricks
Expected Response: databricks-dbrx-instruct

Command: Use Llama
Expected Response: databricks-meta-llama-3-70b-instruct

Command: Use Mixtral
Expected Response: databricks-mixtral-8x7b-instruct

In all other cases respond with databricks-dbrx-instruct.

Only answer with with Databricks model endpoint name. 

Classify this question: {command}
"""

is_question_about_model_prompt = PromptTemplate(
  input_variables= ["command"],
  template = is_question_about_model_str
)

is_about_model_chain = (
    {
        "command": itemgetter("messages") | RunnableLambda(extract_question)
    }
    | is_question_about_model_prompt
    | chat_model
    | StrOutputParser()
)

#Returns Model Endpoint as this is about Databricks: 
print(is_about_model_chain.invoke({
    "messages": [
        {"role": "user", "content": "Use llama3"}
    ]
}))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Use LangChain to retrieve documents from the vector store
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-1.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC Let's add our LangChain retriever. 
# MAGIC
# MAGIC It will be in charge of:
# MAGIC
# MAGIC * Creating the input question embeddings (with Databricks `bge-large-en`)
# MAGIC * Calling the vector search index to find similar documents to augment the prompt with
# MAGIC
# MAGIC Databricks LangChain wrapper makes it easy to do in one step, handling all the underlying logic and API call for you.

# COMMAND ----------

index_name=f"{catalog}.{db}.documentation_vs_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

#Let's make sure the secret is properly setup and can access our vector search index. Check the quick-start demo for more guidance
test_demo_permissions(host, secret_scope="dbdemos", secret_key="rag_sp_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="databricks-bge-large-en")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model, columns=["url"]
    )
    return vectorstore.as_retriever(search_kwargs={'k': 4})

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What is SafetyCulture?"}]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improve document search using LLM to generate a better sentence for the vector store, based on the chat history
# MAGIC
# MAGIC We need to retrieve documents related the the last question but also the history.
# MAGIC
# MAGIC One solution is to add a step for our LLM to summarize the history and the last question, making it a better fit for our vector search query. Let's do that as a new step in our chain:

# COMMAND ----------

# DBTITLE 1,Contextual Query Generation Chain
from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

#Let's try it
output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is SafetyCulture?"}
    ]
})
print(f"Test retriever query without history: {output}")

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is SafetyCulture?"}, 
        {"role": "assistant", "content": "SafetyCulture is a company that provides tooling and applications for workplace safety and sire inspections."}, 
        {"role": "user", "content": "Does it support creating custom templates?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Let's put it together
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-model-2.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC
# MAGIC Let's now merge the retriever and the full LangChain chain.
# MAGIC
# MAGIC We will use a custom LangChain template for our assistant to give a proper answer.
# MAGIC
# MAGIC Make sure you take some time to try different templates and adjust your assistant tone and personality for your requirement.
# MAGIC
# MAGIC

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
You are a trustful assistant for SafetyCulture Application users, customers, as well as SafetyCulture company. You are answering questions about SafetyCulture, SafetyCulture concepts, SafetyCulture people, SafetyCulture Inspections, SafetyCulture company, SafetyCulture Application, or other SafetyCulture topics and other information related to SafetyCulture. If SafetyCulture word appears by itself, assume the question is about the company. If SafetyCulture word is not in the question, assume the question is about SafetyCulture company and SafetyCulture related topics. If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "assistant". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

dbrx_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500, temperature = 0.8, extra_params = {
    "top_p": 0.95
  })
llama_model = ChatDatabricks(endpoint="databricks-meta-llama-3-70b-instruct", max_tokens = 500, temperature = 0.8, extra_params = {
    "top_p": 0.95
  })
mixtral_model = ChatDatabricks(endpoint="databricks-mixtral-8x7b-instruct", max_tokens = 500, temperature = 0.8, extra_params = {
    "top_p": 0.95
  })

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["url"] for d in docs]

branch_node_model = RunnableBranch(
  (lambda x: "databricks-mixtral-8x7b-instruct" in x["model"].lower(), (itemgetter("prompt") | mixtral_model)),
  (lambda x: "databricks-meta-llama-3-70b-instruct" in x["model"].lower(), (itemgetter("prompt") | llama_model)),
  (itemgetter("prompt") | dbrx_model)
)

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | dbrx_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about SafetyCulture.', "sources": []})
)

# branch_node = RunnableBranch(
#   (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
#   (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
#   irrelevant_question_chain
# )

branch_node = relevant_question_chain

full_chain = (
  {
    "question_is_relevant": is_about_safetyculture_chain,
    "model": is_about_model_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history)
  }
  | branch_node
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's try our full chain:

# COMMAND ----------

# DBTITLE 1,Asking an out-of-scope question
import json
non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What is SafetyCulture?"}, 
        {"role": "assistant", "content": "SafetyCulture is a company that provides tooling and applications for workplace safety and sire inspections."}, 
        {"role": "user", "content": "Why is the sky blue?"}
    ]
}
print(f'Testing with a non relevant question...')
response = full_chain.invoke(non_relevant_dialog)
display_chat(non_relevant_dialog["messages"], response)

# COMMAND ----------

# DBTITLE 1,Asking a relevant question
dialog = {
    "messages": [
        {"role": "user", "content": "What is SafetyCulture?"}, 
        {"role": "assistant", "content": "SafetyCulture is a company that provides tooling and applications for workplace safety and sire inspections."}, 
        {"role": "user", "content": "Does it support creating custom templates?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
display_chat(dialog["messages"], response)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Register the chatbot model to Unity Catalog

# COMMAND ----------

import cloudpickle
import langchain
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.advanced-chatbot-model"

with mlflow.start_run(run_name="chatbot_rag") as run:
    #Get our model signature from input/output
    output = full_chain.invoke(dialog)
    signature = infer_signature(dialog, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__
        ],
        input_example=dialog,
        signature=signature,
        example_no_conversion=True,
    )

# COMMAND ----------

client = MlflowClient()
model_version_infos = client.search_model_versions(f"name = '{model_name}'")
new_model_version = max([int(model_version_info.version) for model_version_info in model_version_infos])
client.set_registered_model_alias(model_name, "prod", new_model_version)

# COMMAND ----------

# MAGIC %md Let's try loading our model

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
model.invoke(dialog)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC We've seen how we can improve our chatbot, adding more advanced capabilities to handle a chat history.
# MAGIC
# MAGIC As you add capabilities to your model and tune the prompt, it will get harder to evaluate your model performance in a repeatable way.
# MAGIC
# MAGIC Your new prompt might work well for what you tried to fixed, but could also have impact on other questions.
# MAGIC
# MAGIC ## Next: Introducing offline model evaluation with MLflow
# MAGIC
# MAGIC To solve these issue, we need a repeatable way of testing our model answer as part of our LLMOps deployment!
# MAGIC
# MAGIC Open the next [03-Offline-Evaluation]($./03-Offline-Evaluation) notebook to discover how to evaluate your model.

# COMMAND ----------

# dialog = {
#     "messages": [
#         {"role": "user", "content": "What is SafetyCulture?"}, 
#         {"role": "assistant", "content": "SafetyCulture is a digital workplace software designed to help IT managers, digital workplace managers, HR teams, and employee experience teams manage their teams effectively, automate manual processes, and ensure organizational compliance. It offers customizable digital forms, a central hub for communication, employee training and certification tracking, task progress monitoring, and safety audits, hazard checklists, and incident reporting features. It's available on both mobile app (iOS and Android) and web-based software. SafetyCulture is best for businesses prioritizing safety and health of their employees, applicable across all industries and useful for various sectors, regardless of size. It emphasizes user convenience by offering ready-to-use templates, real-time corrective actions, efficient recordkeeping, and insightful analytics."}, 
#         {"role": "user", "content": "make it rhyme"}
#     ]
# }
# model.invoke(dialog)
