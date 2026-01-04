#!/usr/bin/env python3
"""Custom agent for fine-tuned Qwen3-VL model at /Users/pran-ker/qwen3vl_exl"""

import os
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.utils.output_beautify import typewriter_print


def init_custom_agent():
    """Initialize agent with custom fine-tuned Qwen3-VL model via VLLM"""
    llm_cfg = {
        'model': 'Qwen3-VL-Custom',
        'model_server': 'http://localhost:8000/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'top_p': 0.8,
            'temperature': 0.7,
            'max_tokens': 2000,
            'max_input_tokens': 6500,
            'max_retries': 10,
        }
    }

    tools = [
        'code_interpreter',
    ]

    bot = Assistant(
        llm=llm_cfg,
        function_list=tools,
        name='Custom Qwen3-VL Agent',
        description='Fine-tuned Qwen3-VL model for specialized vision and language tasks'
    )

    return bot


def test_single_query(query: str = 'Hello, can you see this message?'):
    """Test with a single query"""
    bot = init_custom_agent()
    messages = [{'role': 'user', 'content': query}]
    response_plain_text = ''

    print(f"User: {query}\n")
    for response in bot.run(messages=messages):
        response_plain_text = typewriter_print(response, response_plain_text)


def run_cli():
    """Run agent with command-line interface"""
    bot = init_custom_agent()
    messages = []

    print("=" * 60)
    print("Custom Qwen3-VL Agent")
    print("Model: /Users/pran-ker/qwen3vl_exl (via VLLM)")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60)

    while True:
        try:
            query = input('\nUser: ')
            if query.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            if not query.strip():
                continue

            messages.append({'role': 'user', 'content': query})
            response_plain_text = ''

            for response in bot.run(messages=messages):
                response_plain_text = typewriter_print(response, response_plain_text)

            messages.extend(response)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def run_gui():
    """Run agent with web UI"""
    bot = init_custom_agent()
    chatbot_config = {
        'prompt.suggestions': [
            'Analyze this image for me',
            'Help me understand this code',
            'What can you do?',
        ]
    }

    print("Starting Web UI...")
    print("Model: /Users/pran-ker/qwen3vl_exl (via VLLM)")
    print("Access the UI at: http://localhost:7860")

    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test_single_query()
        elif sys.argv[1] == 'cli':
            run_cli()
        elif sys.argv[1] == 'gui':
            run_gui()
        else:
            print("Usage: python3 custom_qwen3vl_agent.py [test|cli|gui]")
            print("  test - Run a single test query")
            print("  cli  - Run command-line interface")
            print("  gui  - Run web UI (default)")
    else:
        run_gui()
