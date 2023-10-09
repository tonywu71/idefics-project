import pytest
from dataloader.dataset_groups.pokemon_cards_ds import PokemonCards


@pytest.fixture
def ds_group() -> PokemonCards:
    return PokemonCards()

def test_load_pokemon_ds(ds_group: PokemonCards):
    """
    Test that the PokemonCards dataset is correctly loaded.
    """
    for split, ds in ds_group.items():
        assert len(ds) > 0, f"Dataset {split} is empty."
