"""
Entry point for the DepenSage web server.

Usage:
    python -m depensage.web.main [--host HOST] [--port PORT]
"""

import argparse
import os
import ssl
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="DepenSage Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--no-ssl", action="store_true", help="Disable HTTPS")
    args = parser.parse_args()

    ssl_kwargs = {}
    if not args.no_ssl:
        cert_dir = os.path.abspath(".secrets")
        cert_file = os.path.join(cert_dir, "cert.pem")
        key_file = os.path.join(cert_dir, "key.pem")

        if os.path.exists(cert_file) and os.path.exists(key_file):
            ssl_kwargs["ssl_certfile"] = cert_file
            ssl_kwargs["ssl_keyfile"] = key_file
            protocol = "https"
        else:
            print(
                f"SSL certificates not found at {cert_dir}/cert.pem and key.pem\n"
                f"Generate with:\n"
                f"  openssl req -x509 -newkey rsa:2048 -keyout {key_file} "
                f"-out {cert_file} -days 365 -nodes "
                f'-subj "/CN=depensage"\n'
                f"Or run with --no-ssl for HTTP (not recommended)."
            )
            sys.exit(1)
    else:
        protocol = "http"

    print(f"Starting DepenSage at {protocol}://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print("Accessible from any device on the local network")

    uvicorn.run(
        "depensage.web.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
