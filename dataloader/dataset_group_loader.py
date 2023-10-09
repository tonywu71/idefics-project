from dataloader.dataset_groups.pokemon_cards_ds import PokemonCards
from dataloader.dataset_groups.newyorker_caption_ds import NewYorkerCaption

DATASET_NAME_TO_LOAD_FUNC = {
    "pokemon_cards": PokemonCards,
    "newyorker_caption": NewYorkerCaption
}
