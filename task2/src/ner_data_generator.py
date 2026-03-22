"""Synthetic BIO-tagged sentence generation for NER training."""

import random

from config import ANIMAL_CLASSES, RANDOM_STATE

# Sentence templates with {animal} placeholder
# The placeholder may appear at the beginning, middle, or end of the sentence
_TEMPLATES = [
    "There is a {animal} in the picture",
    "I can see a {animal}",
    "The {animal} is sitting on the grass",
    "A {animal} was spotted near the river",
    "Look at that {animal} over there",
    "The photo shows a {animal}",
    "This image contains a {animal}",
    "A wild {animal} appeared in the forest",
    "The {animal} is eating food",
    "I found a {animal} in my backyard",
    "Is that a {animal} in the picture",
    "Can you see the {animal} here",
    "A beautiful {animal} is resting in the shade",
    "The large {animal} walks through the savanna",
    "We observed a {animal} at the zoo",
    "A small {animal} hides behind the rock",
    "There are two {animal} playing together",
    "The {animal} runs across the field",
    "I photographed a {animal} during my trip",
    "A friendly {animal} approached us",
    "The {animal} looks directly at the camera",
    "Have you ever seen a {animal} this close",
    "That {animal} is absolutely magnificent",
    "A {animal} sleeps peacefully under the tree",
    "The {animal} jumped over the fence",
    "My favorite animal is the {animal}",
    "A young {animal} follows its mother",
    "The {animal} is standing in the water",
    "I think this is a {animal}",
    "A {animal} was captured on camera",
    "The majestic {animal} roams the plains",
    "Someone left a photo of a {animal} here",
    "A curious {animal} peeks from behind the bush",
    "The {animal} has beautiful markings",
    "I spotted a {animal} during the safari",
]

# Negative templates (no animal mentioned)
_NEGATIVE_TEMPLATES = [
    "There is nothing in the picture",
    "The image is blurry and dark",
    "I cannot see any creature here",
    "This photo shows a beautiful landscape",
    "The picture was taken at sunset",
    "There are only trees and rocks",
    "I see a car parked near the building",
    "The weather looks nice today",
    "This is an empty field",
    "No living creature is visible",
]


def _tokenize_and_tag(sentence: str, animal: str | None) -> dict:
    """Split a sentence into tokens and assign BIO tags.
    B - indicates the beginning of an entity.
    I - indicates a token is contained inside the same entity (for example, the State token is a part of an entity like Empire State Building).
    0 - indicates the token doesn’t correspond to any entity.

    Parameters:
        sentence: str
            Input sentence.
        animal: str | None
            Animal name present in the sentence, or None for negatives.

    Returns:
        dict:
            Dictionary with 'tokens' and 'labels' keys.
    """
    # Split sentence into words and start with all "O" (outside) labels
    # e.g. "There is a cow" -> tokens: ["There", "is", "a", "cow"]
    #                          labels: ["O", "O", "O", "O"]
    tokens = sentence.split()
    labels = ["O"] * len(tokens)

    if animal is not None:
        animal_tokens = animal.split()

        # Find where the animal name appears in the sentence
        # strip punctuation and compare case-insensitively
        for i in range(len(tokens) - len(animal_tokens) + 1):
            match = all(
                tokens[i + j].lower().strip(".,!?") == animal_tokens[j].lower()
                for j in range(len(animal_tokens))
            )
            if match:
                # Tag the first token as B-ANIMAL, the rest as I-ANIMAL
                # e.g. "cow" -> ["B-ANIMAL"]
                labels[i] = "B-ANIMAL"
                for j in range(1, len(animal_tokens)):
                    labels[i + j] = "I-ANIMAL"
                break  # only tag the first occurrence

    return {"tokens": tokens, "labels": labels}


def generate_ner_dataset(
    n_train: int = 3000,
    n_val: int = 500,
) -> tuple[list[dict], list[dict]]:
    """Generate synthetic NER training and validation datasets.
    Each sample is a dictionary with 'tokens' (list of words) and
    'labels' (list of BIO tags).

    Parameters:
        n_train: int
            Number of training samples.
        n_val: int
            Number of validation samples.

    Returns:
        tuple[list[dict], list[dict]]:
            Training and validation datasets.

    Raises:
        ValueError:
            If n_train or n_val is not a positive integer.
    """
    if not isinstance(n_train, int) or n_train < 1:
        raise ValueError("n_train must be a positive integer.")

    if not isinstance(n_val, int) or n_val < 1:
        raise ValueError("n_val must be a positive integer.")

    # Fixed seed so the generated dataset is the same every time
    rng = random.Random(RANDOM_STATE)
    total = n_train + n_val

    samples: list[dict] = []

    for i in range(total):
        # Around 18% negative samples (no animal) — tried 10%, 18%, 30%;
        # 18% gave the best balance between learning animals and learning to say "no animal"
        if rng.random() < 0.18:
            template = rng.choice(_NEGATIVE_TEMPLATES)
            samples.append(_tokenize_and_tag(template, None))
        else:
            # Pick a random template and a random animal, fill in the placeholder
            # e.g. "There is a {animal} in the picture" + "cow"
            #    -> "There is a cow in the picture"
            template = rng.choice(_TEMPLATES)
            animal = rng.choice(ANIMAL_CLASSES)
            sentence = template.format(animal=animal)
            samples.append(_tokenize_and_tag(sentence, animal))

    # Shuffle so training and validation sets have a good mix
    rng.shuffle(samples)

    # Split into training and validation sets
    return samples[:n_train], samples[n_train:]
