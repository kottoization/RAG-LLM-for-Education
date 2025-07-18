import io
from contextlib import redirect_stdout

import gradio as gr

from AgentModule import create_agent

agent = create_agent()

CSS = """
* {
  font-family: 'Segoe UI', Tahoma, sans-serif;
}
#chatbot .message.user {
  background-color: #e6f3ff;
  border-radius: 8px;
}
#chatbot .message.bot {
  background-color: #f0f0f0;
  border-radius: 8px;
}
"""


def respond(message: str, history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = agent.invoke({"input": message})["output"]
    history = history + [(message, result)]
    logs = buffer.getvalue()
    return history, logs


def build_interface() -> gr.Blocks:
    with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# EduGen Chat", elem_id="title")
        chatbot = gr.Chatbot(elem_id="chatbot")
        with gr.Row():
            msg = gr.Textbox(placeholder="Type your message and press enter...", container=False)
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
        logs = gr.Textbox(label="Terminal output", lines=8)

        def clear_history():
            return [], ""

        msg.submit(respond, [msg, chatbot], [chatbot, logs])
        send.click(respond, [msg, chatbot], [chatbot, logs])
        clear.click(clear_history, None, [chatbot, logs])

    return demo


def main() -> None:
    demo = build_interface()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
