import torch
import numpy
import pickle
import transformers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """
    Main method to take user input and output an appropriate set of sound waves
    sound like Weird Al produced by our GAN.
    """

    # Set up tokenizer
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Load from pickle
    with open("./weird_al_gan.p") as file:
        model = pickle.load(file)

    while query != "exit":

        # Get user input as a tensor
        query = input("What are the words you would like Weird Al to say?\n")
        query_tensor = tokenizer.encode(query).to(DEVICE)

        # Generate output given the weird al gan, will be a vetor representing 
        # the fourier transform of the sound waves
        output = model.generate(input_ids=query_tensor)


if __name__ == "__main__":
    main()