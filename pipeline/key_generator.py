import hashlib
import secrets

api_key = secrets.token_hex(32)
print(api_key)

hash_to_store = hashlib.sha256(api_key.encode()).hexdigest()
print(hash_to_store)