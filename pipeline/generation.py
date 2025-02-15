from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set!")
print(f"API Key: {OPENAI_API_KEY[:5]}********")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


class Generate:
    def __init__(self, model, prompt, size, quality, n):
        self.model = model
        self.prompt = prompt
        self.size = size
        self.quality = quality
        self.n = n

    async def generate_image(self):
        response = await client.images.generate(
            model=self.model,
            prompt=self.prompt,
            size=self.size,
            quality=self.quality,
            n=self.n
        )
        return response.data[0].url

    async def generate_image_with_revised_prompt(self):
        response = await client.images.generate(
            model=self.model,
            prompt=self.prompt,
            size=self.size,
            quality=self.quality,
            n=self.n
        )
        return response.data[0].revised_prompt, response.data[0].url

