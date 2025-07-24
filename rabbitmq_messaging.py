import ujson
import pika
import os
import ssl

def rabbitmq_conn(event):
    rabbitmq_host = os.getenv("RABBITMQ_HOST", 'b-452fa475-5934-4f0e-b0ff-8f83e8511da8.mq.eu-north-1.amazonaws.com')
    rabbitmq_port = int(os.getenv("RABBITMQ_PORT", 5671))  # Use 5671 for SSL/TLS
    rabbitmq_username = os.getenv("RABBITMQ_USERNAME", 'eagleai')
    rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", 'Password111#Eagleai')
    rabbitmq_vhost = os.getenv("RABBITMQ_VHOST", "/")

    print(f"Connecting to RabbitMQ ")

    credentials = pika.PlainCredentials(rabbitmq_username, rabbitmq_password)
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    
    parameters = pika.ConnectionParameters(
        host=rabbitmq_host,
        port=rabbitmq_port,
        virtual_host=rabbitmq_vhost,
        credentials=credentials,
        ssl_options=pika.SSLOptions(context=ssl_context)
    )

    try:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        queue = os.getenv("RABBITMQ_API_QUEUE", "EAGLEAI_DEV_API_QUEUE") if 'sender' in event else 'TELEGRAM_BOT'
        channel.queue_declare(queue=queue, durable=False)

        channel.basic_publish(
            exchange='',
            routing_key=queue,
            #body=event
            body=ujson.dumps(event)
        )
        print("Message sent to RabbitMQ queue:", queue)

        connection.close()
    except Exception as e:
        print(f"Error connecting to RabbitMQ: {e}")

# Example usage
# event = {
#     "command": "PRICE",
#     "data": {"price": 123.45}
# }

# rabbitmq_conn(event)