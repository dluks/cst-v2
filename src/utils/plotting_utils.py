from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from src.utils.trait_utils import get_trait_name_from_id


def set_font(font: str) -> None:
    """Set font for matplotlib plots."""
    if font not in ["FreeSans"]:
        raise ValueError("Font not supported yet.")

    if font == "FreeSans":
        # Search for font files on the system
        def find_font_files(font_name: str) -> dict[str, str]:
            """Find font files by searching the system."""
            found_fonts = {}

            # Get all system fonts
            all_fonts = font_manager.findSystemFonts(
                fontext="ttf"
            ) + font_manager.findSystemFonts(fontext="otf")

            for font_file in all_fonts:
                font_path = Path(font_file)
                if font_path.exists():
                    # Check for regular font
                    if (
                        font_name.lower() in font_path.name.lower()
                        and "bold" not in font_path.name.lower()
                    ):
                        found_fonts["regular"] = str(font_path)
                    # Check for bold font
                    elif (
                        font_name.lower() in font_path.name.lower()
                        and "bold" in font_path.name.lower()
                    ):
                        found_fonts["bold"] = str(font_path)

            return found_fonts

        # Find FreeSans fonts
        freesans_fonts = find_font_files("FreeSans")

        if not freesans_fonts:
            raise FileNotFoundError(
                "FreeSans font not found on the system. "
                "Please install the font (e.g., sudo apt-get install fonts-freefont-ttf) "
                "or ensure it's available in your font directories."
            )

        # Add fonts to matplotlib
        if "regular" in freesans_fonts:
            font_manager.fontManager.addfont(freesans_fonts["regular"])

        if "bold" in freesans_fonts:
            font_manager.fontManager.addfont(freesans_fonts["bold"])

        plt.rcParams["font.family"] = "FreeSans"


def show_available_fonts() -> None:
    """Print all available fonts."""
    fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    for font in fonts:
        print(font)


def add_human_readables(df: pd.DataFrame) -> pd.DataFrame:
    """Add human readable trait name and trait set abbreviations."""
    return df.pipe(add_trait_name).pipe(add_trait_set_abbr)


def add_trait_name(df: pd.DataFrame) -> pd.DataFrame:
    trait_id_to_name = {
        trait_id: get_trait_name_from_id(trait_id)[0]
        for trait_id in df.trait_id.unique()
    }
    return df.assign(trait_name=df.trait_id.map(trait_id_to_name))


def add_trait_set_abbr(df: pd.DataFrame) -> pd.DataFrame:
    trait_set_to_abbr = {"splot": "SCI", "splot_gbif": "COMB", "gbif": "CIT"}
    return df.assign(trait_set_abbr=df.trait_set.map(trait_set_to_abbr))
