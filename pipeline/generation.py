class Generate:
    """
    A class to handle image generation using OpenAI's API.
    """
    def __init__(self, client, model, prompt, size, quality, style, n):
        """
        Initializing the Generate class with required parameters.

        :param model: The AI model name to use for generation. ("Dall-E-2", "Dall-E-3")
        :param prompt: The text prompt for image generation.
        :param size: The size of the generated image.
        :param quality: The quality of the generated image. ("standard", "hd")
        :param n: The number of images to generate.
        """
        self.client = client
        self.model = model
        self.prompt = prompt
        self.size = size
        self.quality = quality
        self.style = style
        self.n = n

    async def generate_image(self):
        """
        Generate an image based on the provided prompt and parameters.

        :return: URL of the generated image.
        """
        response = await self.client.images.generate(
            model=self.model,
            prompt=self.prompt,
            size=self.size,
            quality=self.quality,
            style=self.style,
            n=self.n
        )
        print(response.data[0].url)
        return response.data[0].url

    async def generate_image_with_revised_prompt(self):
        """
        Generate an image and retrieve the revised prompt (if modified by the model).

        :return: Tuple containing the revised prompt and the image URL.
        """
        response = await self.client.images.generate(
            model=self.model,
            prompt=self.prompt,
            size=self.size,
            quality=self.quality,
            style=self.style,
            n=self.n
        )
        return response.data[0].revised_prompt, response.data[0].url

