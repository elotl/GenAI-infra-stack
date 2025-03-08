import os

import uvicorn


def main():
    # Check environment variables to determine which app to run
    weaviate_url = os.getenv("WEAVIATE_URI_WITH_PORT")
    weaviate_grpc_url = os.getenv("WEAVIATE_GRPC_URI_WITH_PORT")
    weaviate_index = os.getenv("WEAVIATE_INDEX_NAME")

    host = os.environ.get("APP_HOST", "0.0.0.0")
    port = int(os.environ.get("APP_PORT", "8000"))

    if weaviate_url and weaviate_grpc_url and weaviate_index:
        # Run the CSV to Weaviate app
        print(f"Starting Weaviate app on {host}:{port}")
        uvicorn.run("serverragllm_csv_to_weaviate_local:app", host=host, port=port)
    else:
        # Run the default app
        print(f"Starting default app on {host}:{port}")
        uvicorn.run("serveragllm:app", host=host, port=port)


if __name__ == "__main__":
    main()
