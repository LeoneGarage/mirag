# Databricks notebook source
# MAGIC %pip install mlflow==2.12.1 langchain==0.1.17 databricks-vectorsearch==0.22 databricks-sdk==0.27.0 mlflow[databricks] gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# import gradio package
import gradio as gr
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize

# Custom CSS for SafetyCulture theme
logo_url = "https://i.ibb.co/gvhkpDL/37ba1cff6995353f53a8879424f963f3.jpg"
css = f"""
body {{
    font-family: Arial, sans-serif;
    background-color: #F4F5F7;
    color: #333;
}}
.gradio-container {{
    border: 1px solid #E1E4E8;
    border-radius: 2px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
    background-color: #FFFFFF;
    position: relative;
    background-image: url("{logo_url}");
    background-size: 100px;
}}
.gr-button {{
    background-color: #0070F3;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.3s;
}}
.gr-button:hover {{
    background-color: #005BB5;
}}
.gr-input, .gr-textarea {{
    border: 1px solid #E1E4E8;
    border-radius: 4px;
    padding: 10px;
    font-size: 16px;
}}
.gr-label {{
    font-weight: bold;
    color: #333;
    margin-bottom: 5px;
}}
.app.svelte-182fdeq.svelte-182fdeq {{
  padding: 40px 80px 40px 80px;
  max-width: 100%;
}}
"""

# define a function that receives a name and displays "hello name!"
def greet(question):
  w = WorkspaceClient()

  answer = w.serving_endpoints.query(name = serving_endpoint_name, inputs=[{"query": question}])
  return answer.predictions[0]

# the gradio Interface class
### with:
##fn: the function to wrap a UI around
##inputs: which component(s) to use for the input
##outputs: which component(s) to use for the output

# Apply the custom theme to the Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Welcome to SafetyCulture-themed Gradio App")
    name_input = gr.Textbox(label="Question", lines=2)
    greet_button = gr.Button("Submit")
    greet_output = gr.Textbox(label="Answer", lines=50, min_width=200)
    greet_button.click(fn=greet, inputs=name_input, outputs=greet_output)

# launch the app!
demo.launch(share=True)

# COMMAND ----------

stream = (
  spark.readStream
     .format("cloudFiles")
     .option("cloudFiles.format", "csv")
     .option("cloudFiles.schemaLocation", "dbfs:/empty/schema")
     .option("cloudFiles.useIncrementalListing", "auto")
     .load("dbfs:/empty/files")
     .writeStream
     .option("checkpointLocation", "dbfs:/empty/cp")
     .trigger(processingTime='5 minutes')
     .table("`safety-culture`.empty.dummy_data")
)
