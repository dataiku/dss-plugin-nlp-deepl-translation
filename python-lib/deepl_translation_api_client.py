# -*- coding: utf-8 -*-
"""Module with utility functions to call the DeepL translation API"""

import json
import logging
import re

import requests

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


API_EXCEPTIONS = (requests.HTTPError,)


HTTP_ERRORS = {
    "400": "Bad request. Please check error message and your parameters.",
    "403": "Authorization failed. Please supply a valid auth_key parameter.",
    "404": "The requested resource could not be found.",
    "413": "The request size exceeds the limit.",
    "414": "The request URL is too long. You can avoid this error by using a POST request instead of a GET request.",
    "429": "Too many requests. Please wait and resend your request.",
    "456": "Quota exceeded. The character limit has been reached.",
    "5**": "Internal error",
    "503": "Resource currently unavailable. Try again later.",
    "529": "Too many requests. Please wait and resend your request.",
}


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class DeepLClient:
    def __init__(self, api_key, deepl_url) -> None:
        self.api_key = api_key
        self.deepl_url = deepl_url

    def translate(
        self,
        text,
        target_language,
        source_language=None,
        split_sentences="1",
        preserve_formatting="0",
        formality="default",
    ):
        """
        Translates text.

        For detailed parameter information refer to: https://www.deepl.com/docs-api/translating-text/request/
        """
        response = requests.post(
            url=self.deepl_url,
            data={
                "source_lang": source_language,
                "target_lang": target_language,
                "auth_key": self.api_key,
                "text": text,
                "split_sentences": split_sentences,
                "preserve_formatting": preserve_formatting,
                "formality": formality,
            },
        )

        if "translations" in response.text:
            # Returns text from the response object which is a json string, so no need to dump it into json anymore
            return response.text
        else:
            # Extracts error related information
            error_message = response.text
            deepl_info = HTTP_ERRORS.get(str(response.status_code), "")

            # Handles 5** error special case
            if (deepl_info == "") and (str(response.status_code)[0] == "5"):
                deepl_info = HTTP_ERRORS.get("5**", "")

            user_message = (
                "Encountered the following error while sending an API request to DeepL:"
                + f" Error Code: {response.status_code}"
                + f" DeepL Resolution: {deepl_info}"
                + f" Error message: {error_message}"
            )

            raise requests.HTTPError(user_message)
