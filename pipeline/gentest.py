import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)


class Generate:
    def __init__(self, prompt):
        self.prompt = prompt

    async def generate_image(self):
        response = await client.images.generate(
            model="dall-e-3",
            prompt=self.prompt,
            size="1024x1024",
            quality="hd",
            n=1
        )
        return response.data[0].url


async def main():
    generator = Generate(prompt="A futuristic city skyline with a huge neon-lit tower and flying cars.")
    image_url = await generator.generate_image()
    print(f"Generated image URL: {image_url}")

if __name__ == "__main__":
    asyncio.run(main())
