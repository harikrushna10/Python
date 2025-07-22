import ujson
import redis
import os

def redis_conn(event):
    redis_host = os.getenv(
        "REDIS_HOST", "eagleai-cache-dev.jlze65.ng.0001.eun1.cache.amazonaws.com"
    )
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    
    redis_username = os.getenv("REDIS_USERNAME", "")
    redis_password = os.getenv("REDIS_PASSWORD", "")

    print(redis_host)
    print(redis_port)
    print(redis_username)
    print(ujson.dumps(event))
    r = redis.StrictRedis(host=redis_host, port=redis_port, db=0)
    if redis_password != "" and redis_username != "":
        print("Connecting with username and password")
        r = redis.StrictRedis(
            host=redis_host,
            port=redis_port,
            username=redis_username,
            password=redis_password,
            db=0,
        )

    count = r.publish("TELEGRAM_BOT", ujson.dumps(event))
    if count == 0:
        print("No subscribers for this channel*******")