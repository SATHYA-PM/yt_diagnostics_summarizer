import gradio as gr
import requests

SERVER_URL = "http://127.0.0.1:8000/api/v1/query"


def contact_backend_server(api_key: str, youtube_url: str, question: str):
    if not api_key.strip() or not youtube_url.strip() or not question.strip():
        return "⚠️ All inputs are required to construct pipeline parameters.", "N/A", "N/A", "N/A", "N/A"

    payload = {
        "api_key": api_key,
        "youtube_url": youtube_url,
        "question": question
    }

    try:
        response = requests.post(SERVER_URL, json=payload)
        result = response.json()

        if response.status_code == 200 and result.get("status") == "success":
            metrics = result["metrics"]
            return result["data"], metrics["latency"], metrics["tokens"], metrics["throughput"], metrics["cost"]
        else:
            error_msg = result.get("detail", "Unknown server-side breakdown.")
            return f"❌ Server Error:\n{error_msg}", "Error", "Error", "Error", "Error"

    except Exception as e:
        return f"❌ Failed to reach backend server architecture:\n{str(e)}", "Offline", "Offline", "Offline", "Offline"


# --- GRADIO RUNTIME VIEW DEPLOYMENT ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📺 Decoupled YouTube RAG Dashboard")
    gr.Markdown("Client dashboard connected directly to your FastAPI backend engine.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔑 Secure Authorization Entry")
            api_key_input = gr.Textbox(
                label="Provider Secret Key",
                placeholder="sk-... or gsk_...",
                type="password"
            )

            gr.Markdown("### 🔗 Processing Targets")
            url_input = gr.Textbox(label="YouTube Link", placeholder="https://www.youtube.com/watch?v=...")
            query_input = gr.Textbox(label="Query Directive", placeholder="Ask anything about the transcript...")

            submit_btn = gr.Button("Transmit Request to Server", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Real-Time Engine Results")
            output_text = gr.TextArea(label="Server Output Response", interactive=False, lines=6)

            gr.Markdown("### 📊 Micro-Service Telemetry Diagnostics")
            with gr.Row():
                latency_metric = gr.Textbox(label="Backend Latency", interactive=False)
                token_metric = gr.Textbox(label="Server Token Track", interactive=False)
            with gr.Row():
                speed_metric = gr.Textbox(label="Throughput Velocity", interactive=False)
                cost_metric = gr.Textbox(label="Resource Consumption Cost", interactive=False)

    submit_btn.click(
        fn=contact_backend_server,
        inputs=[api_key_input, url_input, query_input],
        outputs=[output_text, latency_metric, token_metric, speed_metric, cost_metric]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)