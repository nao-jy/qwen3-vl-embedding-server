#!/usr/bin/env python3
"""
Command-line interface for Qwen3-VL-Embedding Server.
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-Embedding Server - OpenAI-compatible multimodal embedding server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )

    args = parser.parse_args()

    print("Starting Qwen3-VL-Embedding Server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print("Features: Multimodal (text + images), OpenAI-compatible API")

    uvicorn.run(
        "qwen3_vl_embedding_server.server:app",
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
