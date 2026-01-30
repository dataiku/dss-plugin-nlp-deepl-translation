# -*- coding: utf-8 -*-
"""Module with classes to format results from the DeepL Translation API"""

import logging
from typing import AnyStr
from typing import Dict

import pandas as pd

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    ErrorHandlingEnum,
    build_unique_column_names,
    generate_unique,
    safe_json_loads,
    move_api_columns_to_end,
)

LANGUAGE_CODE_LABELS = {
    "ACE": "Acehnese",
    "AF": "Afrikaans",
    "AN": "Aragonese",
    "AR": "Arabic",
    "AS": "Assamese",
    "AY": "Aymara",
    "AZ": "Azerbaijani",
    "BA": "Bashkir",
    "BE": "Belarusian",
    "BG": "Bulgarian",
    "BHO": "Bhojpuri",
    "BN": "Bengali",
    "BR": "Breton",
    "BS": "Bosnian",
    "CA": "Catalan",
    "CEB": "Cebuano",
    "CKB": "Kurdish (Central)",
    "CS": "Czech",
    "CY": "Welsh",
    "DA": "Danish",
    "DE": "German",
    "EL": "Greek",
    "EN": "English",
    "EN-GB": "English (British)",
    "EN-US": "English (American)",
    "EO": "Esperanto",
    "ES": "Spanish",
    "ES-419": "Spanish (Latin American)",
    "ET": "Estonian",
    "EU": "Basque",
    "FA": "Persian",
    "FI": "Finnish",
    "FR": "French",
    "GA": "Irish",
    "GL": "Galician",
    "GN": "Guarani",
    "GOM": "Goan Konkani",
    "GU": "Gujarati",
    "HA": "Hausa",
    "HE": "Hebrew",
    "HI": "Hindi",
    "HR": "Croatian",
    "HT": "Haitian Creole",
    "HU": "Hungarian",
    "HY": "Armenian",
    "ID": "Indonesian",
    "IG": "Igbo",
    "IS": "Icelandic",
    "IT": "Italian",
    "JA": "Japanese",
    "JV": "Javanese",
    "KA": "Georgian",
    "KK": "Kazakh",
    "KMR": "Kurdish (Northern)",
    "KO": "Korean",
    "KY": "Kyrgyz",
    "LA": "Latin",
    "LB": "Luxembourgish",
    "LMO": "Lombard",
    "LN": "Lingala",
    "LT": "Lithuanian",
    "LV": "Latvian",
    "MAI": "Maithili",
    "MG": "Malagasy",
    "MI": "Maori",
    "MK": "Macedonian",
    "ML": "Malayalam",
    "MN": "Mongolian",
    "MR": "Marathi",
    "MS": "Malay",
    "MT": "Maltese",
    "MY": "Burmese",
    "NB": "Norwegian Bokmål",
    "NE": "Nepali",
    "NL": "Dutch",
    "OC": "Occitan",
    "OM": "Oromo",
    "PA": "Punjabi",
    "PAG": "Pangasinan",
    "PAM": "Pampanga",
    "PL": "Polish",
    "PRS": "Dari",
    "PS": "Pashto",
    "PT": "Portuguese",
    "PT-BR": "Portuguese (Brazilian)",
    "PT-PT": "Portuguese (European)",
    "QU": "Quechua",
    "RO": "Romanian",
    "RU": "Russian",
    "SA": "Sanskrit",
    "SCN": "Sicilian",
    "SK": "Slovak",
    "SL": "Slovenian",
    "SQ": "Albanian",
    "SR": "Serbian",
    "ST": "Southern Sotho",
    "SU": "Sundanese",
    "SV": "Swedish",
    "SW": "Swahili",
    "TA": "Tamil",
    "TE": "Telugu",
    "TG": "Tajik",
    "TH": "Thai",
    "TK": "Turkmen",
    "TL": "Tagalog",
    "TN": "Tswana",
    "TR": "Turkish",
    "TS": "Tsonga",
    "TT": "Tatar",
    "UK": "Ukrainian",
    "UR": "Urdu",
    "UZ": "Uzbek",
    "VI": "Vietnamese",
    "WO": "Wolof",
    "XH": "Xhosa",
    "YI": "Yiddish",
    "YUE": "Cantonese",
    "ZH": "Chinese",
    "ZH-HANS": "Chinese (Simplified)",
    "ZH-HANT": "Chinese (Traditional)",
    "ZU": "Zulu",
}

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GenericAPIFormatter:
    """
    Generic Formatter class for API responses:
    - initialize with generic parameters
    - compute generic column descriptions
    - apply format_row to dataframe
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        self.input_df = input_df
        self.column_prefix = column_prefix
        self.error_handling = error_handling
        self.api_column_names = build_unique_column_names(input_df, column_prefix)
        self.column_description_dict = {
            v: API_COLUMN_NAMES_DESCRIPTION_DICT[k]
            for k, v in self.api_column_names._asdict().items()
        }

    def format_row(self, row: Dict) -> Dict:
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names, self.error_handling)
        logging.info("Formatting API results: Done.")
        return df


class TranslationAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for translation API responses for the DeepL Translation API.
    Make sure the response is a valid JSON.
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        input_column: AnyStr,
        target_language: AnyStr,
        source_language: AnyStr = None,
        column_prefix: AnyStr = "translation_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.translated_text_column_name = generate_unique(
            f"{input_column}_{target_language.replace('-', '_')}", input_df.columns, prefix=None
        )
        self.detected_language_column_name = generate_unique(
            f"{input_column}_language", input_df.columns, prefix=None
        )
        self.source_language = source_language
        self.input_column = input_column
        self.input_df_columns = input_df.columns
        self.target_language = target_language
        self.target_language_label = LANGUAGE_CODE_LABELS[self.target_language]
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[
            self.translated_text_column_name
        ] = f"{self.target_language_label} translation of the '{self.input_column}' column by DeepL Translate"
        if not self.source_language:
            self.column_description_dict[
                self.detected_language_column_name
            ] = f"Detected language of the '{self.input_column}' column by DeepL Translate"

    def format_row(self, row: Dict) -> Dict:
        """
        Formats raw row with response into final dataframe row.

        Args:
            row: Dict of a single dataframe row with a column corresponding to the response.
            The response takes the form (https://www.deepl.com/docs-api/translating-text/request/):
            {
                "translations": [{
                    "detected_source_language":"EN",
                    "text":"Der Tisch ist grün. Der Stuhl ist schwarz."
                }]
            }

        Returns:
            row: Dict of a single formatted dataframe row
        """
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)

        # Extracts the detected_source_language & text with get statements assuring that in case
        # the response is empty an empty string is returned
        if not self.source_language:
            row[self.detected_language_column_name] = LANGUAGE_CODE_LABELS.get(
                response.get("translations", [{}])[0].get("detected_source_language", ""), ""
            )
        row[self.translated_text_column_name] = response.get("translations", [{}])[0].get(
            "text", ""
        )
        return row
