import weaviate

client = weaviate.connect_to_local(
    host="127.0.0.1",  # Use a string to specify the host
    port=8080,
    grpc_port=50051,
)
print("Connection to weaviate successful:", client.is_ready())
client.close()
