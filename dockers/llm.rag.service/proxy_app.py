import os

import uvicorn

from config import WeaviateSettings


def main():
    # Check environment variables to determine which app to run
    weaviate_settings = WeaviateSettings()

    host = os.environ.get("APP_HOST", "0.0.0.0")
    port = int(os.environ.get("APP_PORT", "8000"))

    if weaviate_settings.is_set():
        # Run the CSV to Weaviate app
        print(f"Starting Weaviate app on {host}:{port}")
        uvicorn.run("serverragllm_csv_to_weaviate_local:app", host=host, port=port)
    else:
        # Run the default app
        print(f"Starting default app on {host}:{port}")
        uvicorn.run("serveragllm:app", host=host, port=port)


if __name__ == "__main__":
    main()
