import gradio as gr
import logging
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import os
import itertools
import requests
from gradio.themes.utils import sizes

serving_endpoint_name = os.getenv('SERVING_ENDPOINT')
logo_url = ""
example_q1 = [""]
example_q2 = [""]
example_q3 = [""]
example_q4 = [""]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
workspace_client = WorkspaceClient()

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."

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
    z-index:1;
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
    color: white;
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
.top_image {{
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 20%;
}}
.svelte-1viwdyg {{
  color: black;
}}
.svelte-1viwdyg:hover {{
  color: white;
}}
.svelte-1ed2p3z h1 {{
  color: black;
}}
.svelte-1ed2p3z p {{
  color: black;
}}
"""


def respond(message, history):
    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    w = workspace_client

    try:
        h = [
            t
            for tt in [
                [
                    {"role": "user", "content": f"{h[0]}"},
                    {"role": "assistant", "content": f"{h[1]}"},
                ]
                for h in history
                if not h[1].startswith(
                    "I cannot answer questions that are not about SafetyCulture"
                )
            ]
            for t in tt
        ]
        q = {"messages": h + [{"role": "user", "content": f"{message}"}]}
        response = w.serving_endpoints.query(name=serving_endpoint_name, inputs=[q])
        response_data = (
            response.predictions[0]["result"]
            + "\n\nSources:\n"
            + "\n".join(set(response.predictions[0]["sources"]))
        )

    except Exception as error:
        response_data = f"ERROR status_code: {type(error).__name__}" + str(error)

    return response_data


theme = gr.themes.Soft(
    text_size=sizes.text_sm,
    radius_size=sizes.radius_sm,
    spacing_size=sizes.spacing_sm,
)


ci = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(
        placeholder=f'<img src="{logo_url}" style="opacity:0.5;">',
        show_label=False,
        container=False,
        show_copy_button=True,
        bubble_full_width=True,
    ),
    textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7),
    description="This chatbot is a demo example of LLM chatbot built on Dataricks platform. <br>This content is provided as a LLM RAG educational example, without support. It is using DBRX, has not been tested and should not be used as production content",
    examples=[example_q1, example_q2, example_q3, example_q4],
    cache_examples=False,
    css=css,
    theme=theme,
    retry_btn=None,
    undo_btn=None,
    autofocus=True,
    clear_btn="Clear",
)
with gr.Blocks(css=css, theme=theme, fill_height=True) as demo:
    gr.HTML(f"<img src='{logo_url}' width='200' height='200'>")
    ci.render()

if __name__ == "__main__":
    demo.launch()